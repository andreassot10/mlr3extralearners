#' @title CLassification Naive Bayes for Texts Learner
#' @author andreassot10
#' @name mlr_learners_classif.textmodel_nb
#'
#' @description
#' A [mlr3::Learnerclassif] implementing textmodel_nb from package
#'   \CRANpkg{quanteda.textmodels}.
#' Calls [quanteda.textmodels::textmodel_nb()].
#'
#' @templateVar id classif.textmodel_nb
#' @template section_dictionary_learner
#'
#' @references
#' <optional>
#'
#' @template seealso_learner
#' @template example
#' @export
LearnerClassifNaiveBayesText = R6::R6Class("LearnerClassifNaiveBayesText",
  inherit = LearnerClassif,

  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = ParamSet$new(list(
        ParamDbl$new(
          id = "smooth", default = 1, lower = -Inf, upper = Inf,
          tags = c("train", "learner_pars")),
        ParamFct$new(
          id = "prior", default = "uniform",
          levels = c("docfreq", "termfreq", "uniform"),
          special_vals = list(FALSE), tags = c("train", "learner_pars")),
        ParamFct$new(
          id = "distribution", default = "multinomial",
          levels = c("Bernoulli", "multinomial"),
          special_vals = list(FALSE), tags = c("train", "learner_pars")),
        ########################################################################
        ParamFct$new("stopwords_language",
          tags = c("train", "predict"),
          levels = c(
            "da", "de", "en", "es", "fi", "fr", "hu", "ir", "it",
            "nl", "no", "pt", "ro", "ru", "sv", "smart", "none")),
        ParamUty$new("extra_stopwords", tags = c("train", "predict"), custom_check = check_character),

        ParamLgl$new("tolower", default = TRUE, tags = c("train", "predict", "dfm")),
        ParamLgl$new("stem", default = FALSE, tags = c("train", "predict", "dfm")),

        ParamFct$new("what",
          default = "word", tags = c("train", "predict", "tokenizer"),
          levels = c("word", "word1", "fasterword", "fastestword", "character", "sentence")),
        ParamLgl$new("remove_punct", default = FALSE, tags = c("train", "predict", "tokenizer")),
        ParamLgl$new("remove_symbols", default = FALSE, tags = c("train", "predict", "tokenizer")),
        ParamLgl$new("remove_numbers", default = FALSE, tags = c("train", "predict", "tokenizer")),
        ParamLgl$new("remove_url", default = FALSE, tags = c("train", "predict", "tokenizer")),
        ParamLgl$new("remove_separators", default = TRUE, tags = c("train", "predict", "tokenizer")),
        ParamLgl$new("split_hyphens", default = FALSE, tags = c("train", "predict", "tokenizer")),

        ParamUty$new("n", default = 2, tags = c("train", "predict", "ngrams"), custom_check = curry(check_integerish, min.len = 1, lower = 1, any.missing = FALSE)),
        ParamUty$new("skip", default = 0, tags = c("train", "predict", "ngrams"), custom_check = curry(check_integerish, min.len = 1, lower = 0, any.missing = FALSE)),

        ParamDbl$new("sparsity",
          lower = 0, upper = 1, default = NULL,
          tags = c("train", "dfm_trim"), special_vals = list(NULL)),
        ParamFct$new("termfreq_type",
          default = "count", tags = c("train", "dfm_trim"),
          levels = c("count", "prop", "rank", "quantile")),
        ParamDbl$new("min_termfreq",
          lower = 0, default = NULL,
          tags = c("train", "dfm_trim"), special_vals = list(NULL)),
        ParamDbl$new("max_termfreq",
          lower = 0, default = NULL,
          tags = c("train", "dfm_trim"), special_vals = list(NULL)),

        ParamFct$new("scheme_df",
          default = "count", tags = c("train", "docfreq"),
          levels = c("count", "inverse", "inversemax", "inverseprob", "unary")),
        ParamDbl$new("smoothing_df", lower = 0, default = 0, tags = c("train", "docfreq")),
        ParamDbl$new("k_df", lower = 0, tags = c("train", "docfreq")),
        ParamDbl$new("threshold_df", lower = 0, default = 0, tags = c("train", "docfreq")),
        ParamDbl$new("base_df", lower = 0, default = 10, tags = c("train", "docfreq")),

        ParamFct$new("scheme_tf",
          default = "count", tags = c("train", "predict", "dfm_weight"),
          levels = c("count", "prop", "propmax", "logcount", "boolean", "augmented", "logave")),
        ParamDbl$new("k_tf", lower = 0, upper = 1, tags = c("train", "predict", "dfm_weight")),
        ParamDbl$new("base_tf", lower = 0, default = 10, tags = c("train", "predict", "dfm_weight")),

        ParamFct$new("return_type", default = "bow", levels = c("bow", "integer_sequence", "factor_sequence"), tags = c("train", "predict")),
        ParamInt$new("sequence_length", default = 0, lower = 0, upper = Inf, tags = c("train", "predict", "integer_sequence"))
      ))$
        add_dep("base_df", "scheme_df", CondAnyOf$new(c("inverse", "inversemax", "inverseprob")))$
        add_dep("smoothing_df", "scheme_df", CondAnyOf$new(c("inverse", "inversemax", "inverseprob")))$
        add_dep("k_df", "scheme_df", CondAnyOf$new(c("inverse", "inversemax", "inverseprob")))$
        add_dep("base_df", "scheme_df", CondAnyOf$new(c("inverse", "inversemax", "inverseprob")))$
        add_dep("threshold_df", "scheme_df", CondEqual$new("count"))$
        add_dep("k_tf", "scheme_tf", CondEqual$new("augmented"))$
        add_dep("base_tf", "scheme_tf", CondAnyOf$new(c("logcount", "logave")))$
        add_dep("scheme_tf", "return_type", CondEqual$new("bow"))$
        add_dep("sparsity", "return_type", CondEqual$new("bow"))$
        add_dep("sequence_length", "return_type", CondAnyOf$new(c("integer_sequence", "factor_sequence")))

      ps$values = list(stopwords_language = "smart", extra_stopwords = character(0), n = 1, scheme_df = "unary", return_type = "bow")
      super$initialize(
        id = "classif.textmodel_nb",
        packages = "quanteda.textmodels",
        # Text feature should be a sparse document-feature matrix (DFM) in data.table/data.frame format
        # The DFM will be converted into a quanteda 'dfm' object, which is the format that the learner demands
        # The DFM can be obtained in data.table format from PipeOpTextVectorizer
        feature_types = "character",
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("twoclass", "multiclass"),
        man = "mlr3extralearners::mlr_learners_classif.textmodel_nb"
      )
    }
  ),


  private = list(
    .train = function(task) {

      fn = task$feature_names
      dt = task$data(cols = fn)

      pars = self$param_set$get_values(tags = "learner_pars")

      y = task$truth()
      x = lapply(dt, function(column) {
        tkn = private$.transform_tokens(column)
        tdm = private$.transform_bow(tkn, trim = TRUE) # transform to BOW (bag of words), return term count matrix
        state = list(
          tdm = quanteda::dfm_subset(tdm, FALSE), # empty tdm so we have vocab of training data
          docfreq = invoke(quanteda::docfreq, .args = c(
            list(x = tdm), # column weights
            rename_list(self$param_set$get_values(tags = "docfreq"), "_df$", "")))
        )
        if (self$param_set$values$return_type == "bow") {
          tdm = private$.transform_tfidf(tdm, state$docfreq)
        } else {
          tdm = private$.transform_integer_sequence(tkn, tdm, state)
        }
        tdm
      })

      x = x[[1]]

      mlr3misc::invoke(quanteda.textmodels::textmodel_nb,
        x = x, y = y, .args = pars)
    },

    .predict = function(task) {

      fn = task$feature_names
      dt = task$data(cols = fn)

      newdata = imap(dt, function(column, colname) {
        state = self$state$colmodels[[colname]]
        if (nrow(dt)) {
          tkn = private$.transform_tokens(column)
          tdm = private$.transform_bow(tkn, trim = TRUE)
          #tdm = rbind(tdm, state$tdm) # make sure all columns occur
          #tdm = tdm[, colnames(state$tdm)] # Ensure only train-time features are passed on

          if (self$param_set$values$return_type == "bow") {
            tdm = private$.transform_tfidf(tdm, state$docfreq)
          } else {
            tdm = private$.transform_integer_sequence(tkn, tdm, state)
          }
        } else {
          tdm
        }
        tdm
      })

      newdata = newdata[[1]]
      newdata =
        quanteda::dfm_match(
          newdata,
          features = quanteda::featnames(self$model$x)
        )

      type = ifelse(self$predict_type == "response", "response", "prob")

      pred = mlr3misc::invoke(predict, self$model,
        newdata = newdata,
        type = ifelse(self$predict_type == "response", "class", "probability"))

      if (self$predict_type == "response") {
        PredictionClassif$new(task = task, response = pred)
      } else {
        PredictionClassif$new(task = task, prob = pred)
      }
    },
    # text: character vector of feature column
    .transform_tokens = function(text) {
      corpus = quanteda::corpus(text)
      # tokenize
      tkn = invoke(quanteda::tokens, .args = c(list(x = corpus), self$param_set$get_values(tags = "tokenizer")))
      invoke(quanteda::tokens_ngrams, .args = c(list(x = tkn), self$param_set$get_values(tags = "ngrams")))
    },
    # tkn: tokenized text, result from `.transform_tokens`
    # trim: TRUE during training: trim infrequent features
    .transform_bow = function(tkn, trim) {
      pv = self$param_set$get_values()
      remove = NULL
      if (pv$stopwords_language != "none") {
        if (pv$stopwords_language == "smart") {
          remove = stopwords::stopwords(source = "smart")
        } else {
          remove = stopwords::stopwords(language = self$param_set$get_values()$stopwords_language)
        }
      }
      remove = c(remove, pv$extra_stopwords)
      # document-feature matrix
      tdm = invoke(quanteda::dfm, .args = c(list(x = tkn, remove = remove), self$param_set$get_values(tags = "dfm")))
      # trim rare tokens
      if (trim) {
        invoke(quanteda::dfm_trim, .args = c(list(x = tdm), self$param_set$get_values(tags = "dfm_trim")))
      } else {
        tdm
      }
    },
    .transform_integer_sequence = function(tkn, tdm, state) {
      # List of allowed tokens:

      pv = insert_named(list(min_termfreq = 0, max_termfreq = Inf), self$param_set$get_values(tags = "dfm_trim"))
      dt = data.table(data.table(feature = names(state$docfreq), frequency = state$docfreq))
      tokens = unname(unclass(tkn))
      dict = attr(tokens, "types")
      dict = setkeyv(data.table(k = dict, v = seq_along(dict)), "k")
      dict = dict[dt][pv$min_termfreq < get("frequency") & get("frequency") < pv$max_termfreq, ]

      # pad or cut x to length l
      pad0 = function(x, l) {
        c(x[seq_len(min(length(x), l))], rep(0, max(0, l - length(x))))
      }

      il = self$param_set$values$sequence_length
      if (is.null(il)) il = max(map_int(tokens, length))
      tokens = map(tokens, function(x) {
        x = pad0(ifelse(x %in% dict$v, x, 0), il)
        data.table(matrix(x, nrow = 1))
      })
      tokens = rbindlist(tokens)
      if (self$param_set$values$return_type == "factor_sequence") {
        tokens = map_dtc(tokens, factor, levels = dict$v[!is.na(dict$v)], labels = dict$k[!is.na(dict$v)])
      }
      as.dfm(tokens)
    },
    .transform_tfidf = function(tdm, docfreq) {
      if (!quanteda::nfeat(tdm)) {
        return(tdm)
      }
      # code copied from quanteda:::dfm_tfidf.dfm (adapting here to avoid train/test leakage)
      x = invoke(quanteda::dfm_weight, .args = c(
        list(x = tdm),
        rename_list(self$param_set$get_values("dfm_weight"), "_tf$", "")))
      v = docfreq
      j = methods::as(x, "dgTMatrix")@j + 1L
      x@x = x@x * v[j]
      x
    }
  )
)

mlr_learners$add("classif.textmodel_nb", LearnerClassifNaiveBayesText)
