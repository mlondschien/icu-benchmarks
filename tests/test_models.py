import numpy as np
import polars as pl
import pytest
from ivmodels.models import AnchorRegression as IVModelsAnchorRegression
from sklearn.preprocessing import OneHotEncoder

from icu_benchmarks.models import AnchorRegression, DataSharedLasso


@pytest.mark.parametrize(
    "X, y, dataset",
    [
        (
            np.array([[1, 8], [3, 4], [5, 6], [7, 2]]),
            np.array([1.1, 2.3, 2.9, 3.5]),
            np.array(["a", "a", 1, 1]),
        ),
        (
            np.random.randn(100, 2),
            np.random.randn(100),
            np.random.choice([1, 2, -1], 100),
        ),
    ],
)
def test_data_shared_lasso(X, y, dataset):
    X_polars = pl.DataFrame(X, schema=["x0", "x1"])
    y_polars = pl.Series(y)

    model1 = DataSharedLasso(alpha=1)
    model1.fit(X, y, dataset=dataset)
    yhat1 = model1.predict(X)

    model2 = DataSharedLasso(alpha=1)
    model2.fit(X_polars, y_polars, dataset=dataset)
    yhat2 = model2.predict(X_polars)

    assert np.allclose(yhat1, yhat2)
    assert np.allclose(model1.coef_, model2.coef_)
    assert np.allclose(model1.intercept_, model2.intercept_)
    assert "_dataset=1_x_x1" in model2.coef_table().index


@pytest.mark.parametrize(
    "X, y, dataset",
    [
        (
            np.array([[1, 2], [1, 1], [3, 4], [5, 6], [7, 8]]),
            np.array([1.1, 2.2, 2.3, 2.9, 3.5]),
            np.array(["a", "a", "b", 1, 1]),
        ),
        (
            np.random.randn(100, 2),
            np.random.randn(100),
            np.random.choice([1, 2, -1], 100),
        ),
    ],
)
def test_anchor_regression(X, y, dataset):
    X_polars = pl.DataFrame(X, schema=["x0", "x1"])
    y_polars = pl.Series(y)

    model1 = AnchorRegression(alpha=2, gamma=3, scale_predictors=False)
    model1.fit(X, y, dataset=dataset)
    yhat1 = model1.predict(X)
    model2 = AnchorRegression(alpha=2, gamma=3, scale_predictors=False)
    model2.fit(X_polars, y_polars, dataset=dataset)
    yhat2 = model2.predict(X_polars)
    assert np.allclose(yhat1, yhat2)
    assert np.allclose(model1.coef_, model2.coef_)
    assert np.allclose(model1.intercept_, model2.intercept_)

    model3 = IVModelsAnchorRegression(gamma=3, alpha=2, fit_intercept=True)
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    Z = encoder.fit_transform(dataset.reshape(-1, 1))
    model3.fit(X=X, y=y, Z=Z)
    yhat3 = model3.predict(X)

    assert np.allclose(yhat3, yhat1)
    assert np.allclose(model1.coef_, model3.coef_)
    assert np.allclose(model1.intercept_, model3.intercept_)
