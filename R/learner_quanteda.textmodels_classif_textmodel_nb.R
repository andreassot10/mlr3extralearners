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
          tags = "train"),
        ParamFct$new(
          id = "prior", default = "uniform",
          levels = c("docfreq", "termfreq", "uniform"),
          special_vals = list(FALSE), tags = "train"),
        ParamFct$new(
          id = "distribution", default = "multinomial",
          levels = c("Bernoulli", "multinomial"),
          special_vals = list(FALSE), tags = "train")
      ))

      super$initialize(
        id = "classif.textmodel_nb",
        packages = "quanteda.textmodels",
        # Text feature should be a sparse document-feature matrix (DFM) in data.table/data.frame format
        # The DFM will be converted into a quanteda 'dfm' object, which is the format that the learner demands
        # The DFM can be obtained in data.table format from PipeOpTextVectorizer
        feature_types = "numeric",
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("twoclass", "multiclass"),
        man = "mlr3extralearners::mlr_learners_classif.textmodel_nb"
      )
    }
  ),


  private = list(
    .train = function(task) {
      pars = self$param_set$get_values(tags = "train")

      y = task$truth()
      x = quanteda::as.dfm(task$data(cols = task$feature_names))

      mlr3misc::invoke(quanteda.textmodels::textmodel_nb,
        x = x, y = y, .args = pars)
    },

    .predict = function(task) {
      newdata = quanteda::as.dfm(task$data(cols = task$feature_names))
      newdata =
        quanteda::dfm_match(
          newdata,
          features = quanteda::featnames(self$model$x)
        )
      type = ifelse(self$predict_type == "response", "response", "prob")

      pred = mlr3misc::invoke(predict, self$model,
        newdata = newdata,
        type = type)

      if (self$predict_type == "response") {
        PredictionClassif$new(task = task, response = pred)
      } else {
        PredictionClassif$new(task = task, prob = pred)
      }
    }
  )
)

lrns_dict$add("classif.textmodel_nb", LearnerClassifNaiveBayesText)
