import gin
import polars as pl
import pyarrow.dataset as ds
from pyarrow.parquet import ParquetDataset

from icu_benchmarks.constants import DATA_DIR, VARIABLE_REFERENCE_PATH

CONTINUOUS_FEATURES = ["mean", "std", "slope", "fraction_nonnull", "all_missing"]
CATEGORICAL_FEATURES = ["mode", "num_nonmissing"]
TREATMENT_INDICATOR_FEATURES = ["num_nonmissing", "any_nonmissing"]
TREATMENT_CONTINUOUS_FEATURES = ["rate"]


@gin.configurable
def load(
    sources: list[str],
    outcome: str,
    split: str | None = None,
    data_dir=None,
    min_hours: int = 4,
    variables: list[str] | None = None,
    categorical_features: list[str] | None = None,
    continuous_features: list[str] | None = None,
    treatment_indicator_features: list[str] | None = None,
    horizons=[8, 24],
):
    """
    Load data as a pandas DataFrame and a numpy array.

    This automatically subsets `X` and `y` based on the split, `min_hours` and the
    `outcome` variable being non-null. The numpy arrays are contiguous.

    Parameters
    ----------
    sources : list of str
        The sources to load data from.
    outcome : str
        The outcome variable. E.g., `"mortality_at_24h"`.
    split : str, optional, default = None
        Either `"train"`, `"val"`, or `"test"`. If `None`, all data is loaded.
    data_dir : str, optional, default = None
        The directory containing the data.
    min_hours : int, optional, default = 4
        Only load observations with "`time_hours` >= min_hours - 1". That is, historical
        features were computed on at least `min_hours` time steps.
    variables : list of str, optional, default = None
        For which variables (e.g., `hr`) to load features. If `None`, all variables from
        the variable reference with `DatasetVersion` not `None` are loaded.
    categorical_features : list of str, optional, default = None
        Which categorical features to load. E.g., `"mode"`. If `None`, all categorical
        features are loaded.
    continuous_features : list of str, optional, default = None
        Which continuous features to load. E.g., `"mean"`. If `None`, all continuous
        features are loaded.
    treatment_indicator_features : list of str, optional, default = None
        Which treatment indicator features to load. E.g., `"num_nonmissing"`. If `None`,
        all treatment indicator features are loaded.
    horizons : list of int, optional, default = [8, 24]
        The horizons for which to load features.

    Returns
    -------
    df : pandas.DataFrame
        The features.
    y : numpy.ndarray
        The outcome.
    """
    filters = (ds.field("time_hours") >= min_hours - 1) & ~ds.field(outcome).is_null()

    if split == "train":
        filters &= ds.field("hash") < 0.7
    elif split == "val":
        filters &= (ds.field("hash") >= 0.7) & (ds.field("hash") < 0.85)
    elif split == "test":
        filters &= ds.field("hash") >= 0.85
    elif split is not None:
        raise ValueError(f"Invalid split: {split}")

    columns = features(
        variables=variables,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        treatment_indicator_features=treatment_indicator_features,
        horizons=horizons,
    )

    data_dir = DATA_DIR if data_dir is None else data_dir
    # Use ParquetDataset to read multiple files without a copy as in pd.concat.
    df = (
        ParquetDataset(
            [data_dir / source / "features.parquet" for source in sources],
            filters=filters,
        )
        .read(columns=columns + [outcome])
        .to_pandas()
    )

    y = df[outcome]
    assert y.isna().sum() == 0

    return df.drop(columns=[outcome]), y.to_numpy().astype("bool")


def features(
    variables=None,
    categorical_features=None,
    continuous_features=None,
    treatment_indicator_features=None,
    treatment_continuous_features=None,
    horizons=[8, 24],
):
    """
    Get variable-feature names.

    Parameters
    ----------
    variables : list of str, optional, default = None
        For which variables (e.g., `hr`) to return feature names. If `None`, all
        variables from the variable reference with `DatasetVersion` not `None` are used.
    categorical_features : list of str, optional, default = None
        For which categorical features to return feature names. E.g., `["mode"]`. If
        `None`, all categorical features are used.
    continuous_features : list of str, optional, default = None
        For which continuous features to return feature names. E.g., `["mean"]`. If
        `None`, all continuous features are used.
    treatment_indicator_features : list of str, optional, default = None
        For which treatment indicator features to return feature names. E.g.,
        `["num_nonmissing"]`. If `None`, all treatment indicator features are used.
    treatment_continuous_features : list of str, optional, default = None
        For which treatment continuous features to return feature names. E.g.,
        `["rate"]`. If `None`, all treatment continuous features are used.

    Returns
    -------
    features : list of str
        The feature names.
    """
    if continuous_features is None:
        continuous_features = CONTINUOUS_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if treatment_indicator_features is None:
        treatment_indicator_features = TREATMENT_INDICATOR_FEATURES
    if treatment_continuous_features is None:
        treatment_continuous_features = TREATMENT_CONTINUOUS_FEATURES

    variable_reference = pl.read_csv(VARIABLE_REFERENCE_PATH, separator="\t")

    features = []

    for row in variable_reference.rows(named=True):
        variable = row["VariableTag"]

        if row["DatasetVersion"] is None:
            continue

        if variables is not None and variable not in variables:
            continue

        if variables is not None:
            variables.remove(row["VariableTag"])

        if row["LogTransform"] is True:
            variable = f"log_{variable}"

        if row["VariableType"] == "static":
            features += [variable]
        elif row["DataType"] == "continuous":
            features += [f"{variable}_ffilled"]
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in continuous_features
                for horizon in horizons
            ]
        elif row["DataType"] == "categorical":
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in categorical_features
                for horizon in horizons
            ]
        elif row["DataType"] == "treatment_ind":
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in treatment_indicator_features
                for horizon in horizons
            ]
        elif row["DataType"] == "treatment_cont":
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in treatment_continuous_features
                for horizon in horizons
            ]
        else:
            raise ValueError(f"Unknown DataType: {row['DataType']}")

    if variables is not None and len(variables) > 0:
        raise ValueError(f"Unknown variables: {variables}")

    return features
