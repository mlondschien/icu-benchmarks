import anchorboosting
import gin
import glum
import icu_features

# From https://github.com/google/gin-config/blob/master/docs/index.md#making-existing-c\
# lasses-or-functions-configurable:
# Note that gin.external_configurable does not modify the existing class or function, so
# only calls resulting from references to the configurable name in configuration
# strings, or calls to the return value of gin.external_configurable, will have
# parameter bindings applied. Direct calls will remain unaffected.
# TLDR: Need to `from icu_benchmarks.gin import GeneralizedLinearRegressor` instead of
# `from glum import GeneralizedLinearRegressor` in order to apply the gin configuration.
GeneralizedLinearRegressor = gin.external_configurable(
    glum.GeneralizedLinearRegressor, module="glum"
)

load = gin.external_configurable(icu_features.load, module="icu_features.load")

AnchorBooster = gin.external_configurable(
    anchorboosting.AnchorBooster, module="anchorboosting"
)
