import polars as pl
from pathlib import Path
from icu_benchmarks.constants import DATA_DIR
import click

@click.command()
@click.option("--data_dir", type=click.Path(exists=True))
def main(data_dir):
    data_dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    for (source, target, filter_) in [
        ("mimic", "mimic-carevue", pl.col("carevue") & pl.col("metavision").is_null()),
        ("mimic", "mimic-metavision", pl.col("metavision") & pl.col("carevue").is_null()),
        ("miiv", "miiv-late", pl.col("anchoryear") > 2012),
        ("aumc", "aumc-early", pl.col("anchoryear") == 2006),
        ("aumc", "aumc-late", pl.col("anchoryear") == 2013),
    ]:
        sta = pl.scan_parquet(data_dir / source / "sta.parquet").filter(filter_)
        sta = sta.collect()

        dyn = pl.scan_parquet(data_dir / source / "dyn.parquet")
        dyn = dyn.filter(pl.col("stay_id").is_in(sta["stay_id"])).collect()

        (data_dir/target).mkdir(parents=True, exist_ok=True)
        sta.write_parquet(data_dir / target / "sta.parquet")
        dyn.write_parquet(data_dir / target / "dyn.parquet")

if __name__ == "__main__":
    main()