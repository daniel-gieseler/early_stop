import math
from mlflow.tracking import MlflowClient
from src.lossmoother import LosSmoother

def get_mlflow_metric(run_id: str, metric_name: str = 'train_mlm_loss/tok') -> list[float]:
    history = sorted(MlflowClient().get_metric_history(run_id, metric_name), key=lambda m: m.timestamp)
    return [m.value for m in history]

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
        self.target_loss, self.target_step = None, target_step
        if reference_run is None:
            return
        try:
            loss_curve = get_mlflow_metric(reference_run)
            smoother = LosSmoother()
            min_losses = [smoother.update(loss) for loss in loss_curve]
            self.target_loss, self.target_step = min_losses[-1], len(min_losses)
        except Exception as e:
            print(f"Could not set reference run {reference_run}: {e}")

    def _dynamic_ema_update(self, loss: float, step: int) -> float:
        alpha = 2 / (max(1, int(self.sqrt_steps_fraction * math.sqrt(step))) + 1)
        self.mean_predicted_loss += alpha * (loss - self.mean_predicted_loss)
        return self.mean_predicted_loss

    def get_diff_prediction(self) -> float:
        if self.target_loss is None:
            return None
        return self.mean_predicted_loss - self.target_loss

    def update(self, loss: float) -> float:
        min_loss = self.lossmoother.update(loss)
        predicted = self.extrapolator.update(min_loss)
        return self._dynamic_ema_update(predicted, self.lossmoother.step)

    def update_batch(self, losses: list[float]) -> list[float]:
        """Vectorized update - returns same results as calling update() in a loop."""
        min_losses = [self.lossmoother.update(loss) for loss in losses]
        predictions = self.extrapolator.update_batch(min_losses)
        return [self._dynamic_ema_update(float(p), i + 1) for i, p in enumerate(predictions)]



