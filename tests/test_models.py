import numpy as np
import polars as pl
import pytest

from icu_benchmarks.models import DataSharedLasso


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4])),
    ],
)
def test_data_shared_lasso(X, y):
    X_polars = pl.DataFrame(X)
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
