import pandas as pd
import numpy as np
from pathlib import Path
from pysr import PySRRegressor
import plotly.graph_objects as go
import sympy as sp
from IPython.display import display, Markdown

from src.lossmoother import LosSmoother
from src.extrapolator import Extrapolator
from src.losstrapolator import LosStrapolator
from src.pysr_optimization import load_curves


def _preprocess_curve(raw_losses: list[float]) -> dict:
    """Smooth raw losses and return preprocessed data dict."""
    smoother = LosSmoother()
    preprocessed = [smoother.update(loss) for loss in raw_losses]
    return {'raw_losses': raw_losses, 'preprocessed_losses': preprocessed, 'target_loss': preprocessed[-1]}


def _last_exceed_step(errors: np.ndarray, epsilon: float, default: int) -> int:
    """Return the last step where error exceeds epsilon, or default if none."""
    exceed_indices = np.where(errors > epsilon)[0]
    return exceed_indices[-1] + 1 if len(exceed_indices) else default


def collect_all_equations(N: int | None = None, run_id: str | None = None):
    """Collect and aggregate all symbolic equations from saved PySRRegressor runs in 'outputs/'.

    Returns:
        pd.DataFrame: DataFrame containing all equations, their corresponding run_ids, features present,
                    and features actually used in each equation.
    """
    outputs_dir = Path(__file__).parent.parent / "outputs"
    
    if run_id is not None:
        run_folders = [outputs_dir / run_id]
    else:
        run_folders = sorted((f for f in outputs_dir.iterdir() if f.is_dir()), key=lambda x: x.name)
        run_folders = run_folders[:-N] if N else run_folders

    summary = []
    for run_folder in run_folders:
        try:
            model = PySRRegressor.from_file(run_directory=str(run_folder))
            summary.append(model.equations_.assign(
                run_id=run_folder.name,
                features_in=[model.feature_names_in_] * len(model.equations_),
                features_used=lambda df: df['sympy_format'].apply(lambda s: s.free_symbols)
            ))
            del model  # Explicitly cleanup to reduce Julia thread issues
        except Exception as e:
            print(f"Could not load model from {run_folder}: {e}")

    return pd.concat(summary, ignore_index=True) if summary else pd.DataFrame()


def evaluate_model(
    curves: list[tuple[str, list[float]]] | None = None,
    models_df: pd.DataFrame | None = None,
    path: str = 'src/runs_data.json',
    total: int = 4300,
    epsilons: list[float] = (0.02, 0.015, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001),
) -> pd.DataFrame:
    """
    Process runs_experiments and create a dataframe with processed loss data.
    """
    model_key = lambda c, e: f'C_{c}_{e}'
    
    if curves is None:
        curves = load_curves(path, total)
    
    runs_data = {run_id: _preprocess_curve(raw_losses) for run_id, raw_losses in curves}

    if models_df is None:
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
        key = model_key(complexity, eq_id)
        print(f"Evaluating {key}")

        for run_id in runs_data:
            extrapolator = Extrapolator(target_step=total, variable_names=row['features_in'], equation=row['lambda_format'])
            losstrapolator = LosStrapolator(extrapolator=extrapolator, target_step=total)
            predicted = losstrapolator.update_batch(runs_data[run_id]['preprocessed_losses'])
            errors = np.abs(np.array(predicted) - runs_data[run_id]['target_loss'])
            runs_data[run_id][key] = {'predicted': predicted, 'errors': errors}
    

    # Build list of (complexity, eq_id, epsilon, avg_step)
    rows = []
    for eq_id, (_, row) in enumerate(latest_model_df.iterrows()):
        complexity = row['complexity']
        key = model_key(complexity, eq_id)
        for epsilon in epsilons:
            steps = [_last_exceed_step(runs_data[rid][key]['errors'], epsilon, total) for rid in runs_data]
            rows.append({'complexity': complexity, 'eq_id': eq_id, 'epsilon': epsilon, 'steps': np.mean(steps)})
    
    results_df = pd.DataFrame(rows)
    
    # Compute avg steps across key epsilons for scoring
    score_df = (
        results_df.query('epsilon in [0.006, 0.004, 0.002]')
        .groupby(['complexity', 'eq_id'], as_index=False)['steps'].mean()
        .assign(label=lambda df: [model_key(c, e) for c, e in zip(df['complexity'], df['eq_id'])])
    )
    latest_model_df = latest_model_df.reset_index(drop=True).assign(eq_id=lambda df: df.index)
    all_equations_df = score_df.merge(
        latest_model_df[['eq_id', 'complexity', 'sympy_format']], on=['complexity', 'eq_id'], how='left'
    )

    # Build per-run dataframe for plotting
    plot_rows = []
    for run_id, data in runs_data.items():
        row = {'run_id': run_id, 'target_step': total, **{k: data[k] for k in ('raw_losses', 'preprocessed_losses', 'target_loss')}}
        for eq_id, r in latest_model_df.iterrows():
            key = model_key(r['complexity'], eq_id)
            row[f'predicted_{key}'] = data[key]['predicted']
        plot_rows.append(row)

    return results_df, pd.DataFrame(plot_rows), all_equations_df



def plot_tradeoff_curve(df):
    """
    Plot the trade-off between early stopping and accuracy.

    Args:
        df (pd.DataFrame): DataFrame with columns: complexity, eq_id, epsilon, steps
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
    
    model_key = lambda c, e: f'C_{c}_{e}'
    complexities = df['complexity'].unique()
    min_c, max_c = complexities.min(), complexities.max()
    green_red = LinearSegmentedColormap.from_list("greenred", ["#2ca02c", "#d62728"])
    norm = Normalize(vmin=min_c, vmax=max_c)

    fig = go.Figure()
    max_y = df['steps'].max()

    for (complexity, eq_id), subset in df.groupby(['complexity', 'eq_id']):
        label = model_key(int(complexity), int(eq_id))
        color = to_hex(green_red(norm(complexity)))
        subset = subset.sort_values('epsilon')
        fig.add_trace(go.Scatter(
            x=subset['epsilon'], y=subset['steps'],
            mode="lines+markers", name=label,
            line=dict(color=color)
        ))

    # Max step line
    all_epsilons = np.sort(df['epsilon'].unique())
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
    for y, name, w, dash in [
        (target_loss, 'Target Loss', 2, 'dot'),
        (target_loss + margin, 'Target + margin', 1, 'dash'),
        (target_loss - margin, 'Target - margin', 1, 'dash'),
    ]:
        fig.add_trace(go.Scatter(x=[minimum_step, target_step], y=[y, y], mode='lines', name=name, line=dict(color="gray", width=w, dash=dash)))

    fig.update_layout(
        title='Losses Comparison',
        xaxis_title='Step', yaxis_title='Loss',
        xaxis_type="log", legend=dict(x=0.9, y=1.25)
    )
    fig.show()



def display_all_equations(all_equations_df):
    """Display all equations as a formatted table."""
    df = all_equations_df.sort_values(["complexity", "eq_id"])
    rows = [f"| {r['label']} | {r['steps']:.1f} | ${sp.latex(r['sympy_format'])}$ |" for _, r in df.iterrows()]
    display(Markdown("## All Equations\n\n| Label | Steps | Equation |\n|---|---|---|\n" + "\n".join(rows)))