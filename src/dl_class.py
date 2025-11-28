import numpy as np

class EarlyStopping:
    def __init__(self,
        final_step: int,
        log_step: float = 0.1,
        warmup: int = 3,
    ):
        self.step_counter: int = 1
        self.step: list[float] = []
        self.loss: list[float] = []
        self.predicted_loss: list[float] = []
        self.log_step: float = log_step
        self.final_step: float = np.log10(final_step)
        self.warmup: int = warmup
    
    def _preprocess(self, step: int, loss: float):
        step = np.log10(step)
        if not self.step:
            self.step.append(step)
            self.loss.append(loss)
        elif step - self.step[-1] > self.log_step:
            self.step.append(step)
            self.loss.append(min(loss, self.loss[-1]))

    def _first_derivative(self, *, tail_fraction: float) -> float:
        """Derivative of loss w.r.t. log-steps using linear regression on tail fraction."""
        steps = self.step  # already in log-space
        losses = self.loss
        
        # Use the last tail_fraction of the data
        n = len(steps)
        window_size = max(2, int(n * tail_fraction))
        
        # Extract window
        x = steps[-window_size:]
        y = losses[-window_size:]
        
        # Linear regression: slope = cov(x,y) / var(x)
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        
        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        var = sum((xi - x_mean) ** 2 for xi in x)
        
        if var < 1e-10:  # avoid division by zero
            return 0.0
        
        return cov / var  # derivative: d(loss)/d(log_step)

    def _equation(self):
        last_loss = self.loss[-1]
        feature_step = self.step[-1]
        target_step = self.final_step
        first_derivative_tail_10pct = self._first_derivative(tail_fraction = 0.10)
        return last_loss * (target_step / feature_step) ** (first_derivative_tail_10pct - 0.275)

    def update(self, loss: float):
        self._preprocess(self.step_counter, loss)
        self.step_counter += 1
        if self.step_counter <= self.warmup:
            predicted_loss = loss
        else:
            predicted_loss = self._equation()
        self.predicted_loss.append(predicted_loss)
        return predicted_loss



                

                




