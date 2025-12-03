import json
import pandas as pd
import numpy as np
import random
from pysr import PySRRegressor

from lossmoother import LosSmoother
from extrapolator import Extrapolator

def _subsample_logspace(length: int, gap: float) -> np.ndarray:
    """
    Subsample steps [1, length] in log10 space with bin width `gap`,
    always including the last step.
    """
    pos = np.arange(length)          # 0, 1, ..., length-1
    logs = np.log10(pos + 1)         # log10 of steps 1..length
    bins = np.floor(logs / gap).astype(int)

    # Traverse positions in reverse so we keep the last step per bin
    rev_pos = pos[::-1]
    _, first_rev_idx = np.unique(bins[rev_pos], return_index=True)

    # Map back to original positions, sort, then convert to 1-based steps
    chosen_steps = np.sort(rev_pos[first_rev_idx]) + 1
    return chosen_steps


def _create_datapoints(length, gap: float, min_step: int, n_targets: int, max_datapoints_per_target: int):
    samples = _subsample_logspace(length, gap)
    feature_samples = samples[samples >= min_step]
    target_samples = samples[-n_targets:]
    
    result = {}
    for target_step in target_samples:
        valid_features = [
            feature_step
            for feature_step in feature_samples
            if feature_step <= target_step
        ]
        if len(valid_features) > max_datapoints_per_target:
            valid_features = random.sample(valid_features, max_datapoints_per_target)
        result[target_step] = valid_features
    
    return result



def create_dataset(path: str = 'src/runs_data.json', total: int = 4300, variable_names: list[str] = [], gap: float = 0.02, min_step: int = 1000, n_targets: int = 5, max_datapoints_per_target: int = 50) -> pd.DataFrame:
    """
    Process runs_experiments and create a dataframe with processed loss data.
    """ 
    with open(path, 'r') as f:
        runs_experiments = json.load(f)
    
    dfs = []
    for run in runs_experiments:
        if len(run['train_loss']) > total:
            run_id = run['run_id']
            raw_losses = run['train_loss'][:total]
            lossmother = LosSmoother()
            preprocessed_losses = [lossmother.update(loss)[1] for loss in raw_losses]
            datapoints = _create_datapoints(len(raw_losses), gap=gap, min_step=min_step, n_targets=n_targets, max_datapoints_per_target=max_datapoints_per_target)
            for target_step, feature_steps in datapoints.items():
                extrapolator = Extrapolator(target_step, variable_names=variable_names)
                features = [extrapolator.calculate_features(loss = l, step = i+1) for i, l in enumerate(preprocessed_losses)]
                selected_features = [features[i-1] for i in feature_steps]
                df = pd.DataFrame({
                    'run_id': run_id,
                    'target_step': target_step,
                    'target_loss': preprocessed_losses[target_step-1],
                    **{name: [feat[i] for feat in selected_features] for i, name in enumerate(variable_names)}
                })
                dfs.append(df)
            
    return pd.concat(dfs, ignore_index=True)



def fit_pysr_model_on_df(X, y, timeout_in_seconds=5):
    """
    Fit a PySRRegressor model on the provided dataframe `df`.
    Assumes df has columns: ['feature_step', 'target_step', 'last_loss', 'first_derivative_ws2', 'target_loss'].
    Returns the fitted model.
    """
    model = PySRRegressor(
        maxsize=22,
        niterations=10_000_000,
        timeout_in_seconds=timeout_in_seconds,
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
    model.fit(X, y)
    return model


def run_pysr_optimization(min_step: int, n_targets: int):
    variable_names = ['delta_steps', 'last_loss', 'derivative_3']
    df = create_dataset(
        path='src/runs_data.json',
        total=4300,
        variable_names=variable_names,
        gap=0.01,
        min_step=1800,
        n_targets=5,
        max_datapoints_per_target=30
    )
    print(f'Number of datapoints: {len(df)}')
    model = fit_pysr_model_on_df(df[variable_names], df[['target_loss']], timeout_in_seconds=60*10)
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