import pytest
from syrupy import SnapshotAssertion
from src.lossmoother import LosSmoother

def test_update_sequence(snapshot: SnapshotAssertion):
    smoother = LosSmoother()
    results = []
    
    test_sequence = [1.0 - i * 0.5 / 999 for i in range(1000)]
    
    for x in test_sequence:
        minimum = smoother.update(x)
        results.append({
            "step": smoother.step,
            "x": round(x, 5),
            "mean": round(smoother.get_smoothed(), 5),
            "minimum": round(minimum, 5)
        })
    
    assert results == snapshot

