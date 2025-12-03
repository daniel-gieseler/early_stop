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
    
    for eq_id, (_, row) in enumerate(latest_model_df.iterrows()):
        complexity = row['complexity']
        print(f"Evaluating C_{complexity}_{eq_id}")
        callable_equation = row['lambda_format']
        variable_names = row['features_in']

        for run_id in runs_data.keys():
            extrapolator = Extrapolator(target_step=total, variable_names=variable_names, equation=callable_equation)
            losstrapolator = LosStrapolator(extrapolator=extrapolator, target_step=total)
            predicted = losstrapolator.update_batch(runs_data[run_id]['preprocessed_losses'])
            errors = np.abs(np.array(predicted) - runs_data[run_id]['target_loss'])
            runs_data[run_id][f'C_{complexity}_{eq_id}'] = {'predicted': predicted, 'errors': errors}
    

    # Build list of (complexity, eq_id, epsilon, avg_step)
    rows = []
    for eq_id, (_, row) in enumerate(latest_model_df.iterrows()):
        complexity = row['complexity']
        for epsilon in epsilons:
            first_exceed_steps = []
            for run_id in runs_data.keys():
                errors = runs_data[run_id][f'C_{complexity}_{eq_id}']['errors']
                exceed_indices = np.where(errors > epsilon)[0]
                step = exceed_indices[-1] + 1 if len(exceed_indices) > 0 else total
                first_exceed_steps.append(step)
            rows.append({'complexity': complexity, 'eq_id': eq_id, 'epsilon': epsilon, 'steps': np.mean(first_exceed_steps)})
    
    results_df = pd.DataFrame(rows)
    # Compute avg steps across epsilons 0.006, 0.004, 0.002 for all equations
    score_df = results_df[results_df['epsilon'].isin([0.006, 0.004, 0.002])].copy()
    score_df = score_df.groupby(['complexity', 'eq_id'])['steps'].mean().reset_index()
    score_df['label'] = score_df.apply(lambda r: f"C_{int(r['complexity'])}_{int(r['eq_id'])}", axis=1)
    latest_model_df = latest_model_df.reset_index(drop=True)
    latest_model_df['eq_id'] = latest_model_df.index
    all_equations_df = score_df.merge(latest_model_df[['eq_id', 'complexity', 'sympy_format']], on=['complexity', 'eq_id'], how='left')

    # Build per-run dataframe for plotting
    plot_rows = []
    for run_id, data in runs_data.items():
        row = {
            'run_id': run_id,
            'raw_losses': data['raw_losses'],
            'preprocessed_losses': data['preprocessed_losses'],
            'target_loss': data['target_loss'],
            'target_step': total,
        }
        for eq_id, r in latest_model_df.iterrows():
            c = r['complexity']
            row[f'predicted_C_{c}_{eq_id}'] = data[f'C_{c}_{eq_id}']['predicted']
        plot_rows.append(row)

    return results_df, pd.DataFrame(plot_rows), all_equations_df


def plot_tradeoff_curve(df):
    """
    Plot the trade-off between early stopping and accuracy.

    Args:
        df (pd.DataFrame): DataFrame with columns: complexity, eq_id, epsilon, steps
    """
    import plotly.graph_objs as go
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex

    complexities = df['complexity'].unique().tolist()
    min_c, max_c = min(complexities), max(complexities)
    green_red = LinearSegmentedColormap.from_list("greenred", ["#2ca02c", "#d62728"])
    norm = Normalize(vmin=min_c, vmax=max_c)

    fig = go.Figure()
    max_y = df['steps'].max()

    for (complexity, eq_id), subset in df.groupby(['complexity', 'eq_id']):
        label = f"C_{int(complexity)}_{int(eq_id)}"
        color = to_hex(green_red(norm(complexity)))
        subset = subset.sort_values('epsilon')
        fig.add_trace(go.Scatter(
            x=subset['epsilon'], y=subset['steps'],
            mode="lines+markers", name=label,
            line=dict(color=color)
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
        xaxis_type="log", yaxis_type="log", legend_title="Equation"
    )
    fig.show()


import plotly.graph_objects as go

import json


def plot_predicted_vs_true_loss(df, label, n_th=0, margin=0.004, minimum_step=100):
    """Plot raw, preprocessed, and predicted losses for a single run."""
    selected_run_id = sorted(df['run_id'].unique())[n_th]
    run = df[df['run_id'] == selected_run_id].iloc[0]

    raw_losses = run['raw_losses'][minimum_step - 1:]
    preprocessed_losses = run['preprocessed_losses'][minimum_step - 1:]
    predicted = run[f'predicted_{label}'][minimum_step - 1:]
    target_loss = run['target_loss']
    target_step = run['target_step']

    X = np.arange(minimum_step, minimum_step + len(raw_losses))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X, y=raw_losses, mode='lines', name='Raw Loss',
        line=dict(color="black", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=X, y=preprocessed_losses, mode='lines', name='Preprocessed Loss',
        line=dict(color="lightgreen", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=X, y=predicted, mode='lines', name=f'Predicted ({label})',
        line=dict(color="red", width=2)
    ))

    # Horizontal lines as traces so they're toggleable in legend
    fig.add_trace(go.Scatter(
        x=[minimum_step, target_step], y=[target_loss, target_loss],
        mode='lines', name='Target Loss',
        line=dict(color="gray", width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=[minimum_step, target_step], y=[target_loss + margin, target_loss + margin],
        mode='lines', name='Target + margin',
        line=dict(color="gray", width=1, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=[minimum_step, target_step], y=[target_loss - margin, target_loss - margin],
        mode='lines', name='Target - margin',
        line=dict(color="gray", width=1, dash="dash")
    ))

    fig.update_layout(
        title=f'Losses Comparison',
        xaxis_title='Step', yaxis_title='Loss',
        xaxis_type="log", legend=dict(x=0.9, y=1.25)
    )
    fig.show()


import sympy as sp
from IPython.display import display, Markdown

def display_all_equations(all_equations_df):
    """Display all equations as a formatted table."""
    df = all_equations_df.sort_values(["complexity", "eq_id"])
    equations_md = "## All Equations\n\n| Label | Steps | Equation |\n|---|---|---|\n"
    for _, row in df.iterrows():
        latex_eq = sp.latex(row['sympy_format'])
        equations_md += f"| {row['label']} | {row['steps']:.1f} | ${latex_eq}$ |\n"
    display(Markdown(equations_md))