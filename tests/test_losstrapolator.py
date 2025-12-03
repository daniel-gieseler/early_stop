import pytest
from syrupy import SnapshotAssertion
from src.extrapolator import Extrapolator
from src.losstrapolator import LosStrapolator


def _get_test_sequence():
    return 1200, [1.0 - i * 0.5 / 999 for i in range(1000)]


def test_update_sequence(snapshot: SnapshotAssertion):
    target_step, losses = _get_test_sequence()
    
    extrapolator = Extrapolator(target_step=target_step)
    losstrapolator = LosStrapolator(extrapolator=extrapolator, target_step=target_step)
    results = []
    for step, loss in enumerate(losses, 1):
        prediction = losstrapolator.update(loss)
        results.append({
            "step": step,
            "loss": round(loss, 5),
            "prediction": round(float(prediction), 5)
        })
    
    assert results == snapshot(name="losstrapolator_sequence")


def test_update_batch_sequence(snapshot: SnapshotAssertion):
    target_step, losses = _get_test_sequence()
    
    extrapolator = Extrapolator(target_step=target_step)
    losstrapolator = LosStrapolator(extrapolator=extrapolator, target_step=target_step)
    predictions = losstrapolator.update_batch(losses)
    results = [
        {
            "step": step,
            "loss": round(loss, 5),
            "prediction": round(float(pred), 5)
        }
        for step, (loss, pred) in enumerate(zip(losses, predictions), 1)
    ]
    
    assert results == snapshot(name="losstrapolator_sequence")

