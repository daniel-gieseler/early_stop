import json
import pandas as pd
from lossmoother import LosSmoother
from extrapolator import Extrapolator
from losstrapolator import LosStrapolator



from pathlib import Path
import pandas as pd
from pysr import PySRRegressor

def collect_all_equations(N: int | None = None):
    """Collect and aggregate all symbolic equations from saved PySRRegressor runs in 'outputs/'.

    Returns:
        pd.DataFrame: DataFrame containing all equations, their corresponding run_ids, features present,
                    and features actually used in each equation.
    """


    outputs_dir = Path("/Users/danielgieseler/Documents/Code/early_stop/outputs")
    run_folders = [f for f in outputs_dir.iterdir() if f.is_dir()]
    run_folders = sorted(run_folders, key=lambda x: x.name)

    summary = []
    run_folders = run_folders[:-N] if N is not None else run_folders
    for run_folder in run_folders:
        try:
            model = PySRRegressor.from_file(run_directory=str(run_folder))
            eqs = model.equations_.copy()
            eqs['run_id'] = run_folder.name
            eqs['features_in'] = [model.feature_names_in_] * len(eqs)
            eqs['features_used'] = eqs['sympy_format'].apply(lambda s: s.free_symbols)
            summary.append(eqs)
            del model  # Explicitly cleanup to reduce Julia thread issues
        except Exception as e:
            print(f"Could not load model from {run_folder}: {e}")
            continue

    if summary:
        all_equations = pd.concat(summary, ignore_index=True)
    else:
        all_equations = pd.DataFrame()
    return all_equations


# df = collect_all_equations()
# df



import numpy as np

def evaluate_model(path: str = 'src/runs_data.json', total: int = 4300, variable_names: list[str] = [],
    epsilons: list[float] = (0.02, 0.015, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001),
) -> pd.DataFrame:
    """
    Process runs_experiments and create a dataframe with processed loss data.
    """ 
    with open(path, 'r') as f:
        runs_experiments = json.load(f)
    
    runs_data = {}
    for run in runs_experiments:
        if len(run['train_loss']) > total:
            run_id = run['run_id']
            raw_losses = run['train_loss'][:total]
            lossmother = LosSmoother()
            preprocessed_losses = [lossmother.update(loss)[1] for loss in raw_losses]
            runs_data[run_id] = {
                'raw_losses': raw_losses,
                'preprocessed_losses': preprocessed_losses,
                'target_loss': preprocessed_losses[-1],
            }


    models_df = collect_all_equations()
    # constraint 1: length of 'features_used' is 3
    # constraint 2: complexity is below 15 or equal to it
    latest_model_df = models_df[
        (models_df['features_used'].apply(len) == 3) & 
        (models_df['complexity'] <= 15)
    ]

    print(f"Evaluating {len(latest_model_df)} models")
    
    for _, row in latest_model_df.iterrows():
        complexity = row['complexity']
        print(f"Evaluating complexity {complexity}")
        callable_equation = row['lambda_format']
        variable_names = row['features_in']

        for run_id in runs_data.keys():
            extrapolator = Extrapolator(target_step=total, variable_names=variable_names, equation=callable_equation)
            losstrapolator = LosStrapolator(extrapolator=extrapolator, target_step=total)
            #
            predicted = losstrapolator.update_batch(runs_data[run_id]['preprocessed_losses'])
            errors = np.abs(np.array(predicted) - runs_data[run_id]['target_loss'])
            runs_data[run_id][f'C_{complexity}'] = {'predicted': predicted, 'errors': errors}
    
    

    # Build list of (complexity, epsilon, avg_step)
    rows = []
    for _, row in latest_model_df.iterrows():
        complexity = row['complexity']
        for epsilon in epsilons:
            first_exceed_steps = []
            for run_id in runs_data.keys():
                errors = runs_data[run_id][f'C_{complexity}']['errors']
                exceed_indices = np.where(errors > epsilon)[0]
                step = exceed_indices[-1] + 1 if len(exceed_indices) > 0 else total
                first_exceed_steps.append(step)
            rows.append({'complexity': complexity, 'epsilon': epsilon, 'steps': np.mean(first_exceed_steps)})
    
    results_df = pd.DataFrame(rows)
    # Keep best performing equation per complexity (highest steps at epsilon 0.004)
    score_df = results_df[results_df['epsilon'] == 0.004].copy()
    best_per_complexity = score_df.loc[score_df.groupby('complexity')['steps'].idxmax()]
    best_complexity_df = best_per_complexity.merge(
        latest_model_df[['complexity', 'sympy_format']].drop_duplicates('complexity'),
        on='complexity', how='left'
    )

    # Build per-run dataframe for plotting
    plot_rows = []
    complexities = latest_model_df['complexity'].unique()
    for run_id, data in runs_data.items():
        row = {
            'run_id': run_id,
            'raw_losses': data['raw_losses'],
            'preprocessed_losses': data['preprocessed_losses'],
            'target_loss': data['target_loss'],
            'target_step': total,
        }
        for c in complexities:
            row[f'predicted_C{c}'] = data[f'C_{c}']['predicted']
        plot_rows.append(row)

    return results_df, pd.DataFrame(plot_rows), best_complexity_df


def plot_tradeoff_curve(df):
    """
    Plot the trade-off between early stopping and accuracy.

    Args:
        df (pd.DataFrame): DataFrame with columns: complexity, epsilon, steps
    """
    import plotly.graph_objs as go
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex

    complexities = df['complexity'].unique().tolist()
    min_c, max_c = min(complexities), max(complexities)
    green_red = LinearSegmentedColormap.from_list("greenred", ["#2ca02c", "#d62728"])
    norm = Normalize(vmin=min_c, vmax=max_c)
    model_colors = {c: to_hex(green_red(norm(c))) for c in complexities}

    fig = go.Figure()
    max_y = df['steps'].max()

    for complexity in complexities:
        subset = df[df['complexity'] == complexity].sort_values('epsilon')
        fig.add_trace(go.Scatter(
            x=subset['epsilon'], y=subset['steps'],
            mode="lines+markers", name=f"complexity {complexity}",
            line=dict(color=model_colors[complexity])
        ))

    # Max step line
    all_epsilons = sorted(df['epsilon'].unique())
    fig.add_trace(go.Scatter(
        x=all_epsilons, y=[max_y] * len(all_epsilons),
        mode="lines", name="max_step",
        line=dict(color="black", width=2, dash="dot")
    ))

    fig.update_layout(
        title="Trade-off: early-stop vs. accuracy",
        xaxis_title="Max Error", yaxis_title="Avg Step",
        xaxis_type="log", yaxis_type="log", legend_title="Complexity"
    )
    fig.show()


import plotly.graph_objects as go

import json


def plot_predicted_vs_true_loss(df, n_th=0, margin=0.004, complexity=7, minimum_step=100):
    """Plot raw, preprocessed, and predicted losses for a single run."""
    selected_run_id = sorted(df['run_id'].unique())[n_th]
    run = df[df['run_id'] == selected_run_id].iloc[0]

    raw_losses = run['raw_losses'][minimum_step - 1:]
    preprocessed_losses = run['preprocessed_losses'][minimum_step - 1:]
    predicted = run[f'predicted_C{complexity}'][minimum_step - 1:]
    target_loss = run['target_loss']
    target_step = run['target_step']

    X = np.arange(minimum_step, minimum_step + len(raw_losses))

    fig = go.Figure()

    # Raw losses
    fig.add_trace(go.Scatter(
        x=X, y=raw_losses, mode='lines', name='Raw Loss',
        line=dict(color="black", width=1)
    ))

    # Preprocessed losses
    fig.add_trace(go.Scatter(
        x=X, y=preprocessed_losses, mode='lines', name='Preprocessed Loss',
        line=dict(color="green", width=2)
    ))

    # Predicted losses
    fig.add_trace(go.Scatter(
        x=X, y=predicted, mode='lines', name=f'Predicted (C{complexity})',
        line=dict(color="red", width=2)
    ))

    # Target loss horizontal line
    fig.add_shape(type="line", x0=minimum_step, x1=target_step, y0=target_loss, y1=target_loss,
                  line=dict(color="gray", width=2, dash="dot"))
    # Margin lines
    fig.add_shape(type="line", x0=minimum_step, x1=target_step, y0=target_loss + margin, y1=target_loss + margin,
                  line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=minimum_step, x1=target_step, y0=target_loss - margin, y1=target_loss - margin,
                  line=dict(color="gray", width=1, dash="dash"))

    fig.update_layout(
        title=f'Losses for run: {selected_run_id}',
        xaxis_title='Step', yaxis_title='Loss',
        xaxis_type="log", legend=dict(x=0.8, y=0.99)
    )
    fig.show()


import sympy as sp
from IPython.display import display, Markdown

def display_best_complexity(best_complexity_df):
    """Display the best equations per complexity as a formatted table."""
    df = best_complexity_df.sort_values("complexity")
    equations_md = "## Best Equations per Complexity\n\n| C | Steps | Equation |\n|---|---|---|\n"
    for _, row in df.iterrows():
        latex_eq = sp.latex(row['sympy_format'])
        equations_md += f"| {row['complexity']} | {row['steps']:.1f} | ${latex_eq}$ |\n"
    display(Markdown(equations_md))