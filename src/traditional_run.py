from pysr import PySRRegressor
from dataset import get_dataset
from features import FEATURE_MARKET as FM
import pandas as pd
import numpy as np
from typing import Iterable, Dict, List
import json

def fit_pysr_model_on_df(df, step_features, features, timeout_in_seconds=5):
    """
    Fit a PySRRegressor model on the provided dataframe `df`.
    Assumes df has columns: ['feature_step', 'target_step', 'last_loss', 'first_derivative_ws2', 'target_loss'].
    Returns the fitted model.
    """
    X = df[step_features + features]
    y = df[['target_loss']]

    model = PySRRegressor(
        maxsize=15,
        niterations=10_000_000,
        timeout_in_seconds=timeout_in_seconds,
        maxdepth=7,
        binary_operators=["+", "*", "-", "/", "pow"],
        unary_operators=[
            "exp",
            "log",
        ],
        precision=16,
        constraints={
            "pow": (9, 4)
        },
        nested_constraints={
            "log": {"log": 0, "pow": 0, "exp": 0},
            "pow": {"log": 0, "pow": 0, "exp": 0},
            "exp": {"log": 0, "pow": 0, "exp": 0},
        }
    )
    model.fit(X, y)
    return model


def compute_model_losses(model_equations, df, step_features, features):
    third_dataset = df[['run_id', 'target_loss'] + step_features + features].copy()
    X = df[step_features + features]
    for _, row in model_equations.iterrows():
        c = row['complexity']
        third_dataset[f'loss_model_{c}'] = row['lambda_format'](X)
        third_dataset[f'error_model_{c}'] = np.abs(third_dataset[f'loss_model_{c}'] - third_dataset['target_loss'])
    return third_dataset

# ---------- Core function you can reuse on your own DataFrame ----------
def threshold_curve_by_model(
    df: pd.DataFrame,
    epsilons: Iterable[float] = (0.05, 0.02, 0.01, 0.005),
    group_col: str = "run_id",
    step_col: str = "feature_step",
    error_prefix: str = "error_model_",
) -> pd.DataFrame:
    """
    For each model error column (starting with `error_prefix`):
      1) Sort within each `group_col` by `step_col` descending.
      2) Take cumulative max of the error.
      3) For each epsilon, find (per group) the first `step_col` where cummax(error) > epsilon.
         If no value exceeds epsilon, take the *last step* checked (not np.nan)!
      4) Average those steps across groups.
    Returns a DataFrame indexed by epsilon with one column per model.
    """
    # Identify model error columns
    error_cols = [c for c in df.columns if c.startswith(error_prefix)]
    if not error_cols:
        raise ValueError(f"No columns start with '{error_prefix}'. Found: {list(df.columns)}")

    # Work on a sorted copy
    work = df.copy()
    work = work.sort_values([group_col, step_col], ascending=[True, False])

    # Cumulative max per group for each error column
    for col in error_cols:
        work[f"cum_{col}"] = work.groupby(group_col, group_keys=False)[col].cummax()

    # Helper: step where cum error crosses epsilon per (group, model)
    def first_cross_step(g: pd.DataFrame, cum_col: str, eps: float) -> float:
        # Use only the group's rows (do not include grouping columns)
        hits = g.loc[g[cum_col] > eps, step_col]
        if not hits.empty:
            return hits.iloc[0]
        elif len(g) > 0:
            # Nothing crossed epsilon: take the last value of step you checked
            return g[step_col].iloc[-1]
        else:
            # Group is empty
            return np.nan

    # Build results: rows = epsilons, cols = model names
    results: Dict[str, List[float]] = {}
    model_names = [c.replace(error_prefix, "model_") for c in error_cols]  # prettier labels

    for col, model_name in zip(error_cols, model_names):
        cum_col = f"cum_{col}"
        avg_steps = []
        for eps in epsilons:
            # Use include_groups=False to silence pandas warning
            per_group_steps = work.groupby(group_col, group_keys=False).apply(
                lambda g, eps=eps: first_cross_step(g, cum_col, eps), include_groups=False
            )
            avg_steps.append(np.nanmean(per_group_steps.values.astype(float)))
        results[model_name] = avg_steps

    out = pd.DataFrame(results, index=list(epsilons))
    out.index.name = "epsilon"
    return out


def run_traditional_run(timeout_in_seconds, features):
    step_features = ['feature_step', 'target_step']
    df = get_dataset({f: FM[f] for f in features})
    #
    model = fit_pysr_model_on_df(df, step_features, features, timeout_in_seconds=timeout_in_seconds)
    #
    third_dataset = compute_model_losses(model.equations_, df, step_features, features)
    curve = threshold_curve_by_model(third_dataset, epsilons=[0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
    
    # Pivot: key = complexity number, value = list of (epsilon, value) tuples
    pivoted = {}
    for col in curve.columns:
        complexity = col.replace("model_", "")
        pivoted[complexity] = [(eps, val) for eps, val in zip(curve.index, curve[col])]
    
    with open(f"outputs/{model.run_id_}/tradeoff_curve.json", "w") as f:
        json.dump(pivoted, f, indent=4)



