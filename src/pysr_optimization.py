import json
import pandas as pd
import numpy as np
import random
from pysr import PySRRegressor

from src.lossmoother import LosSmoother
from src.extrapolator import Extrapolator

def _subsample_logspace(length: int, gap: float) -> np.ndarray:
    """Subsample steps [1, length] in log10 space, keeping last step per bin."""
    steps = np.arange(1, length + 1)
    bins = (np.log10(steps) / gap).astype(int)
    is_last_in_bin = np.append(bins[:-1] != bins[1:], True)
    return steps[is_last_in_bin]


def _create_datapoints(length: int, gap: float, min_step: int, n_targets: int, max_n: int) -> dict:
    """Map each target step to a list of valid feature steps (subsampled, optionally capped)."""
    samples = _subsample_logspace(length, gap)
    features = samples[samples >= min_step]
    targets = samples[-n_targets:]
    
    def select(target):
        valid = features[features <= target]
        return random.sample(list(valid), max_n) if len(valid) > max_n else list(valid)
    
    return {t: select(t) for t in targets}


def load_curves(path: str, total: int, n: int | None = None) -> list[tuple[str, list[float]]]:
    """Load curves from JSON, filtering and truncating to `total` steps."""
    with open(path, 'r') as f:
        runs = json.load(f)
    curves = [
        (run['run_id'], run['train_loss'][:total])
        for run in runs if len(run['train_loss']) > total
    ]
    return curves[:n] if n else curves


def create_dataset(
    curves: list[tuple[str, list[float]]],
    variable_names: list[str] = [],
    gap: float = 0.02,
    min_step: int = 1000,
    n_targets: int = 5,
    max_n: int = 50
) -> pd.DataFrame:
    """Process curves and create a dataframe with processed loss data."""
    dfs = []
    for run_id, raw_losses in curves:
        smoother = LosSmoother()
        losses = [smoother.update(loss) for loss in raw_losses]
        datapoints = _create_datapoints(len(raw_losses), gap, min_step, n_targets, max_n)
        for target, steps in datapoints.items():
            ext = Extrapolator(target, variable_names=variable_names)
            all_features = [ext.calculate_features(loss=l, step=i) for i, l in enumerate(losses, 1)]
            selected = [all_features[s-1] for s in steps]
            dfs.append(pd.DataFrame({
                'run_id': run_id,
                'target_step': target,
                'target_loss': losses[target-1],
                **dict(zip(variable_names, zip(*selected)))
            }))
    return pd.concat(dfs, ignore_index=True)


def run_pysr_optimization(min_step: int, n_targets: int):
    variable_names = ['delta_steps', 'last_loss', 'derivative_3']
    curves = load_curves('src/runs_data.json', total=4300)
    df = create_dataset(
        curves=curves,
        variable_names=variable_names,
        gap=0.01,
        min_step=min_step,
        n_targets=n_targets,
        max_n=30
    )
    print(f'Number of datapoints: {len(df)}')
    model = PySRRegressor(
        maxsize=22,
        niterations=10_000_000,
        timeout_in_seconds=60*10,
        maxdepth=16,
        binary_operators=["+", "*", "-", "/", "pow"],
        unary_operators=["exp", "log"],
        precision=16,
        constraints={
            "pow": (-1, 4),
        },
        nested_constraints={
            "log": {"log": 0, "pow": 0, "exp": 0},
            "pow": {"log": 1, "pow": 0, "exp": 0},
            "exp": {"log": 1, "pow": 0, "exp": 0},
        },
    )
    model.fit(df[variable_names], df[['target_loss']])
    return model


if __name__ == "__main__":
    import itertools
    import os
    random.seed(42)
    #run_pysr_optimization(min_step=1800, n_targets=5)
    min_steps = [1200, 1400, 1600, 1800, 2000, 2400, 3000]
    n_targets = [2, 4, 8]
    # create the combinations, shuffle and select the first N
    combinations = list(itertools.product(min_steps, n_targets))
    random.shuffle(combinations)
    for i, (min_step, n_target) in enumerate(combinations):
        print(f"Running optimization ({i+1}/{len(combinations)})")
        try:
            model = run_pysr_optimization(min_step=min_step, n_targets=n_target)
            model_folder = f"outputs/{model.run_id_}"
            os.makedirs(model_folder, exist_ok=True)
            with open(os.path.join(model_folder, "parameters.json"), "w") as f:
                json.dump({"min_step": min_step, "n_targets": n_target}, f)
        except Exception as e:
            print(f"Error running optimization ({i+1}/{len(combinations)}): {e}")
            continue