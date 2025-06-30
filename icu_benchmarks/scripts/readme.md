The files `n_samples/n_samples.py` and `refit/refit.py` write an `{result_name}_results.csv` file with columns `n_target`, `seed`, `metric`, `cv_value`, `test_value`, `model_idx` and entries from `get_parameters()` (or `get_refit_parameters()`) and `get_predict_kwargs()`.
Optimally, the runs where the results are written to have a `target` tag.

The file `refit/collect.py` "collects" results from `refit/refit.py` and `n_samples/n_samples.py` as in the format above.
It creates a new `target_run`, with tag `summary_run: True` and `sources = ""`.
In that `target_run`, it writes a `{result_name}_results.csv`. This has the same columns as the `{result_name}_result.csv` from the "source runs", plus an additional `"target"` column. However, it only contains the rows corresponding to the optimal `cv_value`.