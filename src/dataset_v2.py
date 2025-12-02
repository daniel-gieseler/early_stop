from lossmoother import LosSmoother
import json
import pandas as pd
import numpy as np
import random

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


def _create_datapoints(length, gap: float, feature_cutoff: float, target_cutoff: float, max_datapoints: int):
    samples = _subsample_logspace(length, gap)
    cutoff = lambda c: int(len(samples) * c)
    pairs = [
        (feature_step, target_step)
        for feature_step in samples[cutoff(feature_cutoff):]
        for target_step in samples[cutoff(target_cutoff):]
        if feature_step < target_step
    ]
    if len(pairs) > max_datapoints:
        pairs = random.sample(pairs, max_datapoints)
    return zip(*pairs)


def create_dataset(path: str = 'src/runs_data.json', total: int = 4300, feature_callables: list[callable] = [], gap: float = 0.02, feature_cutoff: float = 0.87, target_cutoff: float = 0.97, max_datapoints: int = 50) -> pd.DataFrame:
    """
    Process runs_experiments and create a dataframe with processed loss data.
    """ 
    with open(path, 'r') as f:
        runs_experiments = json.load(f)
    
    runs_data = {}
    for run in runs_experiments:
        if len(run['train_loss']) > total:         
            feature_steps, target_steps = _create_datapoints(len(run['train_loss'][:total]), gap=gap, feature_cutoff=feature_cutoff, target_cutoff=target_cutoff, max_datapoints=max_datapoints)
            runs_data[run['run_id']] = {
                'raw_losses': run['train_loss'][:total],
                'feature_steps': feature_steps,
                'target_steps': target_steps,
                'delta_steps': [t - f for t, f in zip(target_steps, feature_steps)],
            }

    # Preprocess losses for each run using LosSmoother and collect target losses
    for run_data in runs_data.values():
        lossmother = LosSmoother()
        run_data['preprocessed_losses'] = [lossmother.update(loss)[1] for loss in run_data['raw_losses']]
        run_data['target_losses'] = [run_data['preprocessed_losses'][t-1] for t in run_data['target_steps']]

    df = pd.DataFrame(
        {'run_id': run_id, 'feature_step': f_step, 'target_step': t_step, 'target_loss': t_loss, 'delta_steps': delta_steps}
        for run_id, run_data in runs_data.items()
        for f_step, t_step, t_loss, delta_steps in zip(run_data['feature_steps'], run_data['target_steps'], run_data['target_losses'], run_data['delta_steps'])
    )

    for feature_fn in feature_callables:
        for idx, row in df.iterrows():
            cutoff_loss = runs_data[row['run_id']]['preprocessed_losses'][:row['feature_step']] # step is already + 1, so it will be inclusded
            df.loc[idx, feature_fn.__name__] = feature_fn(cutoff_loss)

    return df



def last_loss(loss: list) -> float:
    return loss[-1]

def derivative(loss: list, nth_loss: int = 1) -> float:
    last_loss = loss[-1]
    previous_loss_index = None
    for i in range(len(loss)-2, -1, -1): # not check itself
        if loss[i] != last_loss:
            previous_loss_index = i
            nth_loss -= 1
            if nth_loss == 0:
                break
    if previous_loss_index is None:
        return 0
    return (loss[previous_loss_index] - last_loss) / (previous_loss_index - len(loss) + 1)

def derivative_1(loss: list) -> float:
    return derivative(loss, nth_loss=1)

def derivative_2(loss: list) -> float:
    return derivative(loss, nth_loss=2)

def derivative_3(loss: list) -> float:
    return derivative(loss, nth_loss=3)

def simple_create_dataset(path: str = 'src/runs_data.json') -> pd.DataFrame:
    df = create_dataset(path=path, total=4300, feature_callables=[last_loss, derivative_3])
    # Assign an integer for each unique run_id (from 0 to n-1)
    run_id_to_int = {run_id: idx + 1 for idx, run_id in enumerate(sorted(df['run_id'].unique()))}
    # curve_id shoudl acutally be a value that the id of the group by run_id and target_step
    df['curve_id'] = df.groupby(['run_id', 'target_step']).ngroup()
    #df['curve_id'] = df['run_id'].map(run_id_to_int).astype(float)
    return df

def curve_create_dataset(path: str = 'src/runs_data.json') -> pd.DataFrame:
    df = create_dataset(path=path, total=4300, feature_callables=[last_loss, derivative_3], gap=0.01, feature_cutoff=0.3, target_cutoff=0.999999, max_datapoints=1000)
    df['curve_id'] = df.groupby(['run_id', 'target_step']).ngroup()
    return df


if __name__ == "__main__":
    f_steps, t_steps = _create_datapoints(length=4300, gap=0.01, feature_cutoff=0.3, target_cutoff=0.999999, max_datapoints=1000)
    df = simple_create_dataset(path='src/runs_data.json')

    # show me the list and coutn of the column curve_id
    print(len(df['curve_id'].unique()), len(df))
    # Print unique values of feature steps and target steps as built-in ints
    # also order it
    print(sorted(list(set(map(int, f_steps)))))
    print(sorted(list(set(map(int, t_steps)))))

