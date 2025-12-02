import math
import pandas as pd
import json
import numpy as np
from features import FEATURE_MARKET


def _subsample_uniformly(df, col, gap):
    """
    Uniformly subsample rows by uniquely binning `col` with width `gap`.
    """
    key = np.floor(df[col] / gap).astype(int)
    return df.loc[key.drop_duplicates().index]


def _preprocess_run(df):
    df['cum_min_loss'] = df['loss'].cummin()
    #df['log_cum_min_loss'] = df['cum_min_loss'].apply(lambda x: math.log(x, 10))
    df['log_step'] = df['step'].apply(lambda x: math.log(x, 10))
    df = _subsample_uniformly(df, 'log_step', 0.02)
    return df


def preprocess_runs(path: str = 'src/runs_data.json', total: int = 4300) -> pd.DataFrame:
    """
    Process runs_experiments and create a dataframe with processed loss data.
    """ 
    with open(path, 'r') as f:
        runs_experiments = json.load(f)

    preprocessed_dataset = []
    for run in runs_experiments:
        if len(run['train_loss']) > total:
            values_list = [(i+1, l, lr) for i, (l, lr) in enumerate(list(zip(run['train_loss'], run['lr']))[:total])]
            df = pd.DataFrame(values_list, columns=['step', 'loss', 'lr'])
            preprocessed_df = _preprocess_run(df)
            preprocessed_dataset.append({
                'run_id': run['run_id'],
                'steps': preprocessed_df['log_step'].tolist(),
                'losses': preprocessed_df['cum_min_loss'].tolist(),
                'lr': preprocessed_df['lr'].tolist(),
            })
            
    return preprocessed_dataset


def _create_datapoints(length, gap: int, feature_cutoff = 0.4, target_cutoff = 0.1):
    assert gap > 0
    target_ids = list(range(int(length - length * target_cutoff), length, gap))
    feature_ids = list(range(int(length - length * feature_cutoff), length, gap))
    pairs = [(f, t) for f in feature_ids for t in target_ids if f < t]
    return pairs

def create_dataset(preprocessed_dataset: list, gap=1, feature_cutoff=0.4, target_cutoff=0.1):
    dfs = []
    for run in preprocessed_dataset:
        pairs = _create_datapoints(len(run['steps']), gap, feature_cutoff, target_cutoff)
        df = pd.DataFrame({
            'run_id': run['run_id'],
            'feature_step_id': [f for f, _ in pairs],
            'target_step_id': [t for _, t in pairs],
        })
        df['feature_step'] = [run['steps'][f] for f, _ in pairs]
        df['target_step'] = [run['steps'][t] for _, t in pairs]
        df['target_loss'] = [run['losses'][t] for _, t in pairs]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_data(preprocessed_dataset: list, run_id, feature_step_id) -> dict:
    for run in preprocessed_dataset:
        if run['run_id'] == run_id:
            return {
                'steps': run['steps'][:feature_step_id+1],
                'loss': run['losses'][:feature_step_id+1],
            }
    return None




def calculate_features(dataset: pd.DataFrame, preprocessed_dataset: list, features: dict[str, callable]) -> pd.DataFrame:
    for name, fn in features.items():
        for idx, row in dataset.iterrows():
            data = get_data(preprocessed_dataset, row['run_id'], row['feature_step_id'])
            dataset.loc[idx, name] = fn(data)
    return dataset

def get_dataset(features, path: str = 'src/runs_data.json'):
    preprocessed_runs = preprocess_runs(path=path)
    df = create_dataset(preprocessed_runs, gap=4, feature_cutoff=0.4, target_cutoff=0.1)
    dataset = calculate_features(df, preprocessed_runs, features)
    return dataset

# def get_complete_dataset(features, run_order: int, path: str = 'src/runs_data.json'):
#     # this function should create the dtaset with the 
#     run_df = preprocess_runs(path=path)[run_order]
#     for step_id in range(len(run_df['steps'])):
#         data = {
#             'loss': run_df['losses'][:step_id+1],
#         }
#         for name, fn in features.items():
#             dataset.loc[idx, name] = fn(data)

#     dataset = calculate_features(df, preprocessed_runs, features)


if __name__ == "__main__":
    features = {
        'last_loss': FEATURE_MARKET['last_loss'],
        'first_derivative_tail_10pct': FEATURE_MARKET['first_derivative_tail_10pct'],
    }
    
    dataset = get_dataset(features)
    print(f"Dataset shape: {dataset.shape}")
    print(f"\nDataset columns: {dataset.columns.tolist()}")
    print(f"\nFirst few rows:\n{dataset.head()}")
    print(f"\nDataset info:\n{dataset.info()}")
