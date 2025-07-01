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