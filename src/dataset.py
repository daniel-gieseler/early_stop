import math
from itertools import accumulate
import pandas as pd
import json

TOTAL = 4300

def minimum_loss_at(loss_values: list[float]) -> int:
    """
    Find the step at which the minimum loss value occurs.
    """
    minimum_loss = min(loss_values)
    for step, loss in enumerate(loss_values):
        if loss == minimum_loss:
            return (step+1, minimum_loss)

def cumulative_min(nums):
    """
    Returns the cumulative minimum of a list of numbers.
    
    Example:
        cumulative_min([5, 2, 6, 1, 3]) -> [5, 2, 2, 1, 1]
    """
    return list(accumulate(nums, func=min))


def subsample_average_simple(points, subsample_step=0.1, min_points=5):
    """
    points: list[(x, y)], will be sorted by x
    subsample_step: spacing between kept centers AND half-window radius for averaging
    min_points: require at least this many points in the window
    """
    if not points:
        return []

    pts = sorted(points)  # ensure x-sorted
    out = []
    last_x = pts[0][0] - subsample_step  # so first eligible candidate can pass

    for x, _ in pts:
        # enforce spacing between chosen centers
        if x - last_x < subsample_step:
            continue

        # simple window: all points with |xx - x| <= subsample_step
        lo, hi = x - subsample_step, x + subsample_step
        neighbors_y = [yy for xx, yy in pts if lo <= xx <= hi]

        if len(neighbors_y) >= min_points:
            out.append((x, sum(neighbors_y)/len(neighbors_y)))
            last_x = x
        # else: not enough points around -> skip

    return out

def process_runs_experiments(runs_experiments: list, total: int = TOTAL) -> pd.DataFrame:
    """
    Process runs_experiments and create a dataframe with processed loss data.
    """
    results = []
    for run in runs_experiments:
        if len(run['loss']) > total:
            loss_list = run['loss'][:total]
            m = minimum_loss_at(loss_list)
            res = {
                "experiment_name": run['experiment_name'],
                "min_step": m[0],
                "min_loss": m[1],
                "loss": [(s_+1, x) for s_, x in enumerate(loss_list)],
            }
            res["cumulative_min_loss"] = [(s_+1, x) for s_, x in enumerate(cumulative_min(loss_list))]
            res["log_log_loss"] = [(math.log(s, 10), math.log(x, 10)) for s, x in res["cumulative_min_loss"]]
            res["subsampled_loss"] = subsample_average_simple(res['log_log_loss'], subsample_step=0.02, min_points=1)
            results.append(res)

    df = pd.DataFrame(results)
    df["lr"] = df["experiment_name"].str.extract(r"(\d+(?:\.\d+)?)(?=xlr\b)").astype(int)
    df = df.sort_values(by='lr')
    return df

def get_dataset(path: str):
    with open(path, 'r') as f:
        runs_experiments = json.load(f)

    for run in runs_experiments:
        run['loss'] = sorted(run['loss'], key=lambda x: x[0])
        run['loss'] = [x[2] for x in run['loss']]
    
    df = process_runs_experiments(runs_experiments)
    return df['subsampled_loss'].tolist()