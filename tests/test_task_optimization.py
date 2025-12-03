import random
import pytest
from syrupy import SnapshotAssertion
from src.pysr_optimization import load_curves
from src.task_optimization import collect_all_equations, evaluate_model


def test_evaluate_model(snapshot: SnapshotAssertion):
    random.seed(42)
    curves = load_curves('src/runs_data.json', total=4300, n=2)
    models_df = collect_all_equations(run_id='20251202_190128_hcAuBF')
    
    results_df, plot_df, all_equations_df = evaluate_model(
        curves=curves,
        models_df=models_df,
        total=4300,
        epsilons=[0.01, 0.004],
    )
    
    assert results_df.round(5).to_dict(orient='records') == snapshot(name="results_df")
    assert plot_df.drop(columns=['raw_losses', 'preprocessed_losses']).round(5).to_dict(orient='records') == snapshot(name="plot_df")
    assert all_equations_df.drop(columns=['sympy_format']).round(5).to_dict(orient='records') == snapshot(name="all_equations_df")

