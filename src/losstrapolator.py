from lossmoother import LosSmoother
from mlflow.tracking import MlflowClient

import math

def get_mlflow_metric(
    run_id: str,
    metric_name: str = 'train_mlm_loss/tok'
) -> list[float]:
    client = MlflowClient()
    measurements = [(measurement.timestamp, measurement.value) for measurement in client.get_metric_history(run_id, metric_name)]
    measurements.sort(key=lambda x: x[0])
    return [measurement[1] for measurement in measurements]


class LosStrapolator:
    def __init__(self,
        extrapolator,
        target_step: int,
        reference_run: str = None,
        sqrt_steps_fraction: float = 0.5,
    ):
        self.sqrt_steps_fraction = sqrt_steps_fraction
        self.extrapolator = extrapolator
        self.lossmoother = LosSmoother()
        self.mean_predicted_loss = 0.0
        self._set_target(target_step, reference_run)

    def _set_target(self, target_step: int, reference_run: str | None):
        self.target_loss = None
        self.target_step = target_step
        if reference_run is not None:
            try:
                loss_curve = get_mlflow_metric(reference_run)
                lossmother = LosSmoother()
                min_losses = [lossmother.update(loss)[1] for loss in loss_curve]
                self.target_loss = min_losses[-1]
                self.target_step = len(min_losses)
            except Exception as e:
                print(f"Could not set reference run {reference_run}: {e}")

    def _dynamic_ema_update(self, loss: float):
        N = max(1, math.floor(self.sqrt_steps_fraction * math.sqrt(self.lossmoother.step)))
        alpha = 2 / (N + 1)
        self.mean_predicted_loss = alpha * loss + (1 - alpha) * self.mean_predicted_loss

    def update(self, loss: float):
        _, min_loss = self.lossmoother.update(loss)
        predicted_loss = self.extrapolator.update(min_loss, self.lossmoother.step)
        self._dynamic_ema_update(predicted_loss)
        return self.mean_predicted_loss

    def update_batch(self, losses: list[float]) -> list[float]:
        """Vectorized update - returns same results as calling update() in a loop."""
        # Step 1: Run losses through smoother sequentially (cheap, updates state)
        min_losses, steps = [], []
        for loss in losses:
            _, min_loss = self.lossmoother.update(loss)
            min_losses.append(min_loss)
            steps.append(self.lossmoother.step)
        
        # Step 2: Batch extrapolation (expensive part - now vectorized)
        raw_predictions = self.extrapolator.update_batch(min_losses, steps)
        
        # Step 3: Apply EMA sequentially (cheap)
        results = []
        for pred in raw_predictions:
            self._dynamic_ema_update(float(pred))
            results.append(self.mean_predicted_loss)
        return results



