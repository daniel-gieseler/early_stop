import math

class LosSmoother:
    """
    - Uses only losses seen so far in training.
    - Skips anomalous spikes.
    - Tracks tendency with EMA on a dynamic window.
    - Tracks historical minimum.

    Args:
        sqrt_steps_fraction (float): Fraction of sqrt steps to use as EMA window.
        z_thresh (float): Spike threshold in stds above EMA mean.
    """
    def __init__(self, sqrt_steps_fraction: float = 0.5, z_thresh: float = 2.5):
        self.sqrt_steps_fraction = sqrt_steps_fraction
        self.z_thresh = z_thresh
        self.step = 0
        self.mean, self.var = 0.0, 0.0
        self.minimum = math.inf

    def _dynamic_window(self):
        return max(1, int(self.sqrt_steps_fraction * math.sqrt(self.step)))

    def _ema_update(self, x, N):
        alpha = 2 / (N + 1)
        self.mean = alpha * x + (1 - alpha) * self.mean
        self.var  = alpha * (x - self.mean) ** 2 + (1 - alpha) * self.var

    def _z_score(self, x):
        return 0 if self.var == 0 else (x - self.mean) / math.sqrt(self.var)

    def get_smoothed(self) -> float:
        return self.mean

    def update(self, x: float) -> float:
        self.step += 1
        if self._z_score(x) < self.z_thresh:
            self._ema_update(x, self._dynamic_window())
            self.minimum = min(self.minimum, self.mean)
        return self.minimum