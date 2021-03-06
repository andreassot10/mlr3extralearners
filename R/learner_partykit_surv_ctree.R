#' @title Survival Conditional Inference Tree Learner
#' @author adibender
#' @name mlr_learners_surv.ctree
#'
#' @template class_learner
#' @templateVar id surv.ctree
#' @templateVar caller ctree
#'
#' @references
#' Hothorn T, Zeileis A (2015).
#' “partykit: A Modular Toolkit for Recursive Partytioning in R.”
#' Journal of Machine Learning Research, 16(118), 3905-3909.
#' \url{http://jmlr.org/papers/v16/hothorn15a.html}
#'
#' Hothorn T, Hornik K, Zeileis A (2006).
#' “Unbiased Recursive Partitioning: A Conditional Inference Framework.”
#' Journal of Computational and Graphical Statistics, 15(3), 651–674.
#' \doi{10.1198/106186006x133933}
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerSurvCTree = R6Class("LearnerSurvCTree",
  inherit = LearnerSurv,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamFct$new("teststat",
            levels = c("quadratic", "maximum"),
            default = "quadratic", tags = "train"),
          ParamFct$new("splitstat",
            levels = c("quadratic", "maximum"),
            default = "quadratic", tags = "train"),
          ParamLgl$new("splittest", default = FALSE, tags = "train"),
          ParamFct$new("testtype",
            levels = c(
              "Bonferroni", "MonteCarlo",
              "Univariate", "Teststatistic"), default = "Bonferroni",
            tags = "train"),
          ParamUty$new("nmax", tags = "train"),
          ParamDbl$new("alpha",
            lower = 0, upper = 1, default = 0.05,
            tags = "train"),
          ParamDbl$new("mincriterion",
            lower = 0, upper = 1, default = 0.95,
            tags = "train"),
          ParamDbl$new("logmincriterion", tags = "train"),
          ParamInt$new("minsplit", lower = 1L, default = 20L, tags = "train"),
          ParamInt$new("minbucket", lower = 1L, default = 7L, tags = "train"),
          ParamDbl$new("minprob", lower = 0, default = 0.01, tags = "train"),
          ParamLgl$new("stump", default = FALSE, tags = "train"),
          ParamLgl$new("lookahead", default = FALSE, tags = "train"),
          ParamLgl$new("MIA", default = FALSE, tags = "train"),
          ParamInt$new("nresample",
            lower = 1L, default = 9999L,
            tags = "train"),
          ParamDbl$new("tol", lower = 0, tags = "train"),
          ParamInt$new("maxsurrogate",
            lower = 0L, default = 0L,
            tags = "train"),
          ParamLgl$new("numsurrogate", default = FALSE, tags = "train"),
          ParamInt$new("mtry",
            lower = 0L, special_vals = list(Inf),
            default = Inf, tags = "train"),
          ParamInt$new("maxdepth",
            lower = 0L, special_vals = list(Inf),
            default = Inf, tags = "train"),
          ParamLgl$new("multiway", default = FALSE, tags = "train"),
          ParamInt$new("splittry", lower = 0L, default = 2L, tags = "train"),
          ParamLgl$new("intersplit", default = FALSE, tags = "train"),
          ParamLgl$new("majority", default = FALSE, tags = "train"),
          ParamLgl$new("caseweights", default = FALSE, tags = "train"),
          ParamUty$new("applyfun", tags = "train"),
          ParamInt$new("cores",
            special_vals = list(NULL), default = NULL,
            tags = "train"),
          ParamLgl$new("saveinfo", default = TRUE, tags = "train"),
          ParamLgl$new("update", default = FALSE, tags = "train"),
          ParamFct$new("splitflavour",
            default = "ctree",
            levels = c("ctree", "exhaustive"), tags = c("train", "control")),
          ParamUty$new("offset", tags = "train"),
          ParamUty$new("cluster", tags = "train"),
          ParamUty$new("scores", tags = "train"),
          ParamLgl$new("doFit", default = TRUE, tags = "train"),
          ParamInt$new("maxpts", default = 25000L, tags = c("train", "pargs")),
          ParamDbl$new("abseps", default = 0.001, lower = 0, tags = c("train", "pargs")),
          ParamDbl$new("releps", default = 0, lower = 0, tags = c("train", "pargs"))
        )
      )
      ps$add_dep("nresample", "testtype", CondEqual$new("MonteCarlo"))

      super$initialize(
        id = "surv.ctree",
        packages = c("partykit", "coin", "sandwich", "pracma"),
        feature_types = c("integer", "numeric", "factor", "ordered"),
        predict_types = c("distr", "crank"),
        param_set = ps,
        properties = "weights",
        man = "mlr3extralearners::mlr_learners_surv.ctree"
      )
    }
  ),

  private = list(
    .train = function(task) {
      pars = self$param_set$get_values(tags = "train")

      if ("weights" %in% task$properties) {
        pars$weights = task$weights$weight
      }

      pars_pargs = self$param_set$get_values(tags = "pargs")
      pars$pargs = mlr3misc::invoke(mvtnorm::GenzBretz, pars_pargs)
      pars = pars[!(names(pars) %in% names(pars_pargs))]

      mlr3misc::invoke(partykit::ctree,
        formula = task$formula(),
        data = task$data(), .args = pars)
    },

    .predict = function(task) {

      newdata = task$data(cols = task$feature_names)
      p = mlr3misc::invoke(predict, self$model, type = "prob", newdata = newdata)

      # Define WeightedDiscrete distr6 distribution from the survival function
      x = lapply(p, function(z) data.frame(x = z$time, cdf = 1 - z$surv))
      distr = distr6::VectorDistribution$new(
        distribution = "WeightedDiscrete",
        params = x,
        decorators = c("CoreStatistics", "ExoticStatistics"))

      # Define crank as the mean of the survival distribution
      crank = vapply(x, function(z) sum(z[, 1] * c(z[, 2][1], diff(z[, 2]))), numeric(1))

      list(crank = crank, distr = distr)
    }
  )
)

lrns_dict$add("surv.ctree", LearnerSurvCTree)
