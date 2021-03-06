#' @title Boosted Generalized Linear Classification Learner
#' @author be-marc
#' @name mlr_learners_classif.glmboost
#'
#' @template class_learner
#' @templateVar id classif.glmboost
#' @templateVar caller glmboost
#'
#' @references
#' Bühlmann P, Yu B (2003).
#' “Boosting With the L2 Loss.”
#' Journal of the American Statistical Association, 98(462), 324–339.
#' \doi{10.1198/016214503000125}
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerClassifGLMBoost = R6Class("LearnerClassifGLMBoost",
  inherit = LearnerClassif,
  public = list(

    #' @description
    #' Create a `LearnerClassifGLMBoost` object.
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamDbl$new(
            id = "offset", default = NULL, special_vals = list(NULL),
            tags = "train"),
          ParamFct$new(
            id = "family", default = c("Binomial"),
            levels = c("Binomial", "AdaExp", "AUC"), tags = "train"),
          ParamFct$new(
            id = "link", default = "logit",
            levels = c("logit", "probit"), tags = "train"),
          ParamFct$new(
            id = "type", default = "adaboost",
            levels = c("glm", "adaboost"), tags = "train"),
          ParamLgl$new(id = "center", default = TRUE, tags = "train"),
          ParamInt$new(id = "mstop", default = 100, tags = "train"),
          ParamDbl$new(id = "nu", default = 0.1, tags = "train"),
          ParamFct$new(
            id = "risk", default = "inbag",
            levels = c("inbag", "oobag", "none"), tags = "train"),
          ParamUty$new(id = "oobweights", default = NULL, tags = "train"),
          ParamLgl$new(id = "trace", default = FALSE, tags = "train"),
          ParamUty$new(id = "stopintern", default = FALSE, tags = "train"),
          ParamUty$new(id = "na.action", default = na.omit, tags = "train"),
          ParamUty$new(id = "contrasts.arg", tags = "train")
        )
      )
      ps$add_dep("link", "family", CondEqual$new("Binomial"))
      ps$add_dep("type", "family", CondEqual$new("Binomial"))
      ps$add_dep("oobweights", "risk", CondEqual$new("oobag"))

      super$initialize(
        id = "classif.glmboost",
        packages = "mboost",
        feature_types = c("integer", "numeric", "factor", "ordered"),
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("weights", "twoclass"),
        man = "mlr3extralearners::mlr_learners_classif.glmboost"
      )
    }
  ),

  private = list(
    .train = function(task) {

      # Default family in mboost::glmboost is not useable for twoclass

      if (is.null(self$param_set$values$family)) {
        self$param_set$values = insert_named(
          self$param_set$values,
          list(family = "Binomial"))
      }

      pars = self$param_set$get_values(tags = "train")
      pars_boost = pars[which(names(pars) %in%
        formalArgs(mboost::boost_control))]
      pars_glmboost = pars[which(names(pars) %in%
        formalArgs(mboost::gamboost))]
      pars_binomial = pars[which(names(pars) %in%
        formalArgs(mboost::Binomial))]

      f = task$formula()
      data = task$data()

      if ("weights" %in% task$properties) {
        pars_glmboost = insert_named(
          pars_glmboost,
          list(weights = task$weights$weight))
      }

      pars_glmboost$family = switch(pars_glmboost$family,
        Binomial = invoke(mboost::Binomial, .args = pars_binomial),
        AdaExp = mboost::AdaExp(),
        AUC = mboost::AUC())

      # Predicted probabilities refer to the last factor level
      if (self$predict_type == "prob") {
        levs = c(task$negative, task$positive)
        data[[task$target_names]] = factor(data[[task$target_names]], levs)
      }

      ctrl = invoke(mboost::boost_control, .args = pars_boost)
      invoke(mboost::glmboost, f,
        data = data, control = ctrl,
        .args = pars_glmboost)
    },

    .predict = function(task) {
      family = self$param_set$values$family
      newdata = task$data(cols = task$feature_names)

      if (self$predict_type == "prob" &&
        (family == "AdaExp" || family == "AUC")) {
        stopf("The selected family %s does not support probabilities", family)
      }

      if (self$predict_type == "response") {
        p = invoke(predict, self$model, newdata = newdata, type = "class")
        list(response = p)
      } else {
        p = invoke(predict, self$model, newdata = newdata, type = "response")
        p = matrix(c(p, 1 - p), ncol = 2L, nrow = length(p))
        colnames(p) = task$class_names
        list(prob = p)
      }
    }
  )
)

lrns_dict$add("classif.glmboost", LearnerClassifGLMBoost)
