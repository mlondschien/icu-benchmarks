import numpy as np
import polars as pl
import pytest
from ivmodels.models import AnchorRegression as IVModelsAnchorRegression
from sklearn.preprocessing import OneHotEncoder

from icu_benchmarks.models import AnchorRegression, DataSharedLasso


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4])),
    ],
)
def test_data_shared_lasso(X, y):
    X_polars = pl.DataFrame(X, schema=["x0", "x1"])
    y_polars = pl.Series(y)

    datasets = np.array(["a", "a", 1, 1])

    model1 = DataSharedLasso(alpha=1)
    model1.fit(X, y, datasets=datasets)
    yhat1 = model1.predict(X)

    model2 = DataSharedLasso(alpha=1)
    model2.fit(X_polars, y_polars, datasets=datasets)
    yhat2 = model2.predict(X_polars)

    assert np.allclose(yhat1, yhat2)
    assert np.allclose(model1.coef_, model2.coef_)
    assert np.allclose(model1.intercept_, model2.intercept_)
    assert "_dataset=1_x_x1" in model2.coef_table().index


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1.1, 2.3, 2.9, 3.5])),
    ],
)
def test_anchor_regression(X, y):
    X_polars = pl.DataFrame(X, schema=["x0", "x1"])
    y_polars = pl.Series(y)

    datasets = np.array(["a", "a", 1, 1])

    model1 = AnchorRegression(alpha=2, gamma=2, scale_predictors=False)
    model1.fit(X, y, datasets=datasets)
    yhat1 = model1.predict(X)

    model2 = AnchorRegression(alpha=2, gamma=2, scale_predictors=False)
    model2.fit(X_polars, y_polars, datasets=datasets)
    yhat2 = model2.predict(X_polars)

    assert np.allclose(yhat1, yhat2)
    assert np.allclose(model1.coef_, model2.coef_)
    assert np.allclose(model1.intercept_, model2.intercept_)

    model3 = IVModelsAnchorRegression(gamma=2, alpha=2, fit_intercept=True)
    Z = OneHotEncoder(sparse_output=False).fit_transform(datasets.reshape(-1, 1))
    model3.fit(X=X, y=y, Z=Z)
    yhat3 = model3.predict(X)

    assert np.allclose(yhat3, yhat1)
    assert np.allclose(model1.coef_, model3.coef_)
    assert np.allclose(model1.intercept_, model3.intercept_)
