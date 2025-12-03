import random
import pytest
from syrupy import SnapshotAssertion
from src.pysr_optimization import load_curves, create_dataset


def test_create_dataset(snapshot: SnapshotAssertion):
    random.seed(42)
    curves = load_curves('src/runs_data.json', total=4300, n=2)
    variable_names = ['delta_steps', 'last_loss', 'derivative_3']
    
    df = create_dataset(
        curves=curves,
        variable_names=variable_names,
        gap=0.02,
        min_step=1000,
        n_targets=3,
        max_n=10
    )
    
    result = df.round(5).to_dict(orient='records')
    assert result == snapshot(name="create_dataset")
