This repository was used to create the tables and figures of "Domain Generalization and Domain Adaptation in Intensive Care with Anchor Regression and Boosting" by Malte Londschien, Manuel Burger, Gunnar Rätsch, und Peter Bühlmann.

The main models used are [anchorboosting](https://github.com/mlondschien/anchorboosting) and [ivmodels](https://github.com/mlondschien/ivmodels).

All computations were done on the ETH Euler slurm cluster.
It would be unreasonable to run the experiments on a laptop.

This repository relies heavily on [MLflow](https://mlflow.org/), an open-source solution akin to wandb.

## `icu_benchmarks/scripts`

Scripts to start an experiment and to submit batch jobs (`*/submit.py`), collect an experiment's results (`*/collect.py`), and the actual code to run as part of the jobs (`train/train.py`, `refit/refit.py`, and `n_samples/n_samples.py`).
Also, scripts to create plots (`plots/*`) and to start an mlflow server (`mlflow_server.py`).

### `icu_benchmarks/scripts/train`

Code to run the domain generalization experiments, including leave-one-environment-out CV.
The `submit.py` will start by submitting a job for an MLflow server and creating an experiment `experiment_name` in MLflow.
All jobs will write to the MLflow database via this server.
This reduces concurrency issues.

Given a set of source datasets, it will, for each subset `subset` of size 4, 5, and 6, create folders `logs/{experiment_name}/{'_'.join(subset)}/`, place a `config.gin` and a `command.sh` therein.
It will then run `sbatch .../command.sh` for each folder.
These call `python .../train.py --config .../config.gin`, creating a run that is logged to the MLflow experiment via the MLflow server.
The `slurm.out` is written to the same directory.
The `train.py` runs experiments, and finally writes a `results.csv` to the MLflow job's artifacts.
It also logs the active `config.gin`, the pickled models, the preprocessor, and a `models.json` with parameters.

The `collect.py` will collect and aggregate the `results.csv` of different jobs in one experiment.
It will create a new job with tag `summary_run=True` and write the aggregate to `cv_results.csv`.

### `icu_benchmarks/scripts/refit`

Code to refit models.
Similar to `train`, but results are written directly to the `summary_run=True` run.

### `icu_benchmarks/scripts/n_samples`

Code to fit models on increasingly large subsets of datasets.
Similar to `train`.
Each `submit.py` creates a new experiment (like `train`) and each job corresponds to a new run (like `train`).

### `icu_benchmarks/scripts/plots`

Scripts to create figures. All figures are written to the `summary_run=True` run of an mlflow experiment called `plots`.

 - `plot_cv_results.py`: Plot OOD performance by Anchor Regression / Anchor Boosting's gamma. Supply `--config configs/plots/by_gamma/xyz.gin`.
 - `plot_by_num_samples.py`: Plot performance of OOD-only ("zero-shot"), refit ("few-shot") and a model trained on data from the target domain, by the number of target samples available. Supply `--config configs/plots/num_samples/xyz.gin`. The `{result_name}.csv` files need to be generated first using the `collect.py` scripts.
- `plot_regimes.py`: Generate plot comparing regime transitions i -> ii and ii -> iii. Supply `--config configs/plots/regimes/xyz.gin`. The `{result_name}.csv` files need to be generated first using the `collect.py` scripts.
- `plot_outcomes.py`: Generate Figure 1.
- `plot_residuals_by_gamma.py`: Plot "rescaled" version of `plot_cv_results`. Supply `--config configs/plots/rescaled/xyz.gin`.
- `illustrate_regime_changes.py`: Generate Figure 2.

### `icu_benchmakrs/scripts/tables`

Scripts to generate tables.

- `anchors.py`: Generate Table 1.
- `dataset_summaries.py`: Generate Table 2.

### `icu_benchmarks/mlflow_server.py`

Create an MLflow server independently of an experiment.
Write a file `.mlflow_server` with commands to run on a local machine to forward a port, and url to access the MLflow GUI.
See `slurm_utils.setup_mlflow_server` for details.

## `configs/`

Contains `gin` configuration files for experiment runs (`configs/train/`, `configs/refit`, and `configs/n_samples`) and plots (`configs/plots/`).

## Step-by-step reproduction for [1]'s figures

### Set up data

- Install Burger et al. (2025)'s version of `ricu`. The environment can be set up with conda. Install [dependencies](https://github.com/conda-forge/r-ricu-feedstock/blob/999b2dbf7afb380a0ab8ecbd3230cdeafbcf0f21/recipe/meta.yaml#L43-L57) with conda, `git clone` Burger et al. (2025)'s repository, then install it with `R CMD INSTALL .`.
- Download the 8 Physionet datasets and AUMCdb into `some/folder/ricu-data`. Make sure to download the correct version mentioned in the `ricu/inst/extdata/config/data-sources/dataset.json`'s `"url"` entry.
- Before starting `R`, `export RICU_DATA_PATH=some/folder/ricu-data/`. Then run `ricu::import_src("mimic")`.
- Then run `Rscript base_cohort_extraction.R --ricu_path 'some/folder/ricu-data/' --src 'mimic' --out_path 'some/folder/data/`
- Repeat, replacing `mimic` with `eicu`, `hiric`, `aumc`, `miiv`, `sic`, `nwicu`, `picdb`, `zigong`.
- Set up a new conda environment. Install [`icu-features`](https://github.com/eth-mds/icu-features)' [dependencies](https://github.com/eth-mds/icu-features/blob/main/environment.yml), `git clone` the `icu-features` repo, and install it with `pip install --no-deps icu-features`. Also `pip install icd-mappings`.
- Run `python icu_features/split_datasets.py --data_dir "some/folder/data"`
- Run `python icu_features/icd_codes.py --data_dir "some/folder/data/" --dataset "mimic-carevue"` and then `python icu_features/feature_engineering.py --dataset "mimic-carevue" --data_dir "some/folder/data/"`
- Repeat, replacing `mimic-carevue` with `eicu`, `hiric`, `aumc`, `miiv`, `sic`, `nwicu`, `picdb`, `zigong`. See also `icu-features`' [integration tests](https://github.com/eth-mds/icu-features/blob/f900ff014bfb2dc3034a078003c6355bbca7bb57/.github/workflows/ci.yaml#L8-L41).

All of this will be computationally expensive and might require up to 128 GB of memory. Consider using a cluster.

### Generate fully OOD plots

The setup below works on a cluster managed by slurm.

- Install `icu-benchmarks`' [dependencies](https://github.com/mlondschien/icu-benchmarks/blob/main/environment.yml).
- Set up your first `mlflow server`, e.g., with `python icu_benchmarks/scripts/mlflow_server --tracking_uri sqlite:////some/path/mlruns.db --artifact_location some/path/artifacts`. We strongly recommend using the `sqlite` backend. This will print a command of the form `ssh euler -L <local_port>:<ip>:<port> -N &` to be run on a local machine. The MLflow UI can be accessed at `localhost:<local_port>`. See the docstring of `setup_mlflow_server` for details.
- Run `python icu_benchmarks/scripts/train/submit.py --experiment_name "crea_anchor" --outcome log_creatinine_in_24h" --artifact_location "some/path/artifacts" --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/train/anchor.gin`. This will submit one job for a mlflow server, then 22 jobs to train linear anchor models.
- Run `python icu_benchmarks/scripts/train/submit.py --experiment_name "crea_algbm" --outcome log_creatinine_in_24h" --artifact_location "some/path/artifacts" --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/train/lgbm_anchor_regression.gin`. This will submit one job for a mlflow server, then 22 jobs to train boosted anchor models.
- Once these runs have completed, run `python icu_benchmarks/scripts/plots/plot_cv_results.py --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/plots/by_gamma/crea_anchor_colored.gin` and `python icu_benchmarks/scripts/plots/plot_cv_results.py --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/plots/by_gamma/crea_algbm.gin`. These will create a new experiment `plots` with a single run. Open this run in the mlflow UI, go to artifacts. You will find figures 1 and 2 under `paper/crea_algbm.pdf` and `paper/crea_anchor_colored`.
- Repeat above for the `log_lactate_in_4h` outcome. For the outcomes `kidney_failure_in_48h` and `circulatory_failure_in_8h`, use the `lgbm_anchor_classification` `gin-config`. To create the anchor boosting tuning plots, use the corresponding `configs/train/lgbm_anchor_..._tuning.gin` and `configs/plots/by_gamma/..._tuning.gin`. To create the plots with varying anchors, also run `python icu_benchmarks/train/submit.py ... --anchor_formula "icd10_blocks only"`, `python icu_benchmarks/train/submit.py ... --anchor_formula "patient_id"`, `python icu_benchmarks/train/submit.py ... "bs(year, 4) + C(ward) + C(insurance) + C(adm)"` and then run `python icu_benchmarks/plots/plot_cv_results.py ... --config configs/by_gamma/..._which_anchor.gin`. Adapt the `experiment_name` to match those in the `config/plots/by_gamma/....gin` or adapt the `experiment_name` keys below `"line"` in the `...gin`.
- Next, we reproduce the plots with "number of patients" on the x-axis. Run
  - `python icu_benchmarks/n_samples/submit.py --experiment_name crea_glm_n_samples --outcome log_creatinine_in_24h --artifact_location "some/path/artifacts" --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/n_samples/glm.gin`
  - `python icu_benchmarks/n_samples/submit.py --experiment_name crea_lgbm_n_samples --outcome log_creatinine_in_24h --artifact_location "some/path/artifacts" --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/n_samples/lgbm.gin`
  -  `python icu_benchmarks/refit/submit.py --experiment_name crea_anchor --artifact_location "some/path/artifacts" --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/refit_linear.gin`
  -  `python icu_benchmarks/refit/submit.py --experiment_name crea_algbm --artifact_location "some/path/artifacts" --tracking_uri "sqlite:////some/path/mlruns.db" --config configs/refit_lgbm.gin`.
- Once these finished, run
  - `python icu_benchmarks/scripts/n_samples/collect.py --experiment_name lact_glm_n_samples --result_name n_samples --tracking_uri "sqlite:////some/path/mlruns.db"`
  -  `python icu_benchmarks/scripts/n_samples/collect.py --experiment_name lact_lgbm_n_samples --result_name n_samples --tracking_uri "sqlite:////some/path/mlruns.db"`
  -   `python icu_benchmarks/scripts/refit/collect.py --result_name refit_linear --experiment_name lact_anchor --tracking_uri "sqlite:////some/path/mlruns.db"`
  -    `python icu_benchmarks/scripts/refit/collect.py --result_name refit_linear --experiment_name lact_anchor --tracking_uri "sqlite:////some/path/mlruns.db" --gamma_1`
  -    `python icu_benchmarks/scripts/refit/collect.py --result_name refit_linear --experiment_name lact_algbm --tracking_uri "sqlite:////some/path/mlruns.db"`
  -   `python icu_benchmarks/scripts/refit/collect.py --result_name refit_linear --experiment_name lact_algbm --tracking_uri "sqlite:////some/path/mlruns.db" --gamma_1`
  -    `python icu_benchmarks/scripts/train/collect.py --result_name cv_results --experiment_name lact_algbm --tracking_uri "sqlite:////some/path/mlruns.db"`
  -    `python icu_benchmarks/scripts/train/collect.py --result_name cv1_results --experiment_name lact_algbm --tracking_uri "sqlite:////some/path/mlruns.db" --gamma_1`
  -   `python icu_benchmarks/scripts/train/collect.py --result_name cv_results --experiment_name lact_anchor --tracking_uri "sqlite:////some/path/mlruns.db"`
  -   `python icu_benchmarks/scripts/train/collect.py --result_name cv1_results --experiment_name lact_anchor --tracking_uri "sqlite:////some/path/mlruns.db" --gamma_1`
- Now, create the plots for lactate regression by running
  - `python icu_benchmarks/plots/by_n_samples.py --tracking_uri "sqlite:////some/path/mlruns/mlruns.db --config configs/plots/lact_algbm.gin` and
  - `python icu_benchmarks/plots/by_n_samples.py  --tracking_uri "sqlite:////some/path/mlruns/mlruns.db --config configs/plots/lact_anchor.gin`.
- Repeat for the outcomes `log_creatinine_in_24h`, `kidney_failure_in_48h`, and `circulatory_failure_in_8h`.