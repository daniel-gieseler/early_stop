import numpy as np

def delta_steps(target_step: int, step: int) -> int:
    return target_step - step

def last_loss(loss: list) -> float:
    return loss[-1]

def derivative(loss: list, nth_loss: int = 1) -> float:
    last_loss = loss[-1]
    previous_loss_index = None
    for i in range(len(loss)-2, -1, -1): # not check itself
        if loss[i] != last_loss:
            previous_loss_index = i
            nth_loss -= 1
            if nth_loss == 0:
                break
    if previous_loss_index is None:
        return 0
    return (loss[previous_loss_index] - last_loss) / (previous_loss_index - len(loss) + 1)

class Extrapolator:
    def __init__(self,
        target_step: int,
        variable_names: list[str],
        equation: callable = lambda x: x[0],
    ):
        self.equation = equation
        self.variable_names = variable_names
        self.target_step = target_step
        self.losses: list[float] = []

    def calculate_features(self, loss, step):
        self.losses.append(loss)
        features = []
        for v in self.variable_names:
            if v == 'delta_steps':
                features.append(delta_steps(self.target_step, step))
            elif v == 'last_loss':
                features.append(last_loss(self.losses))
            elif v == 'derivative_3':
                features.append(derivative(self.losses, nth_loss=3))
            else:
                raise ValueError(f"Unknown variable: {v}")
        return features

    def update(self, loss: float, step: int):
        features = self.calculate_features(loss, step)
        return self.equation(np.array([features]))

    def update_batch(self, losses: list[float], steps: list[int]) -> np.ndarray:
        """Vectorized update - returns same results as calling update() in a loop."""
        n = len(losses)
        losses_arr = np.array(losses)
        features_matrix = []
        
        for v in self.variable_names:
            if v == 'delta_steps':
                features_matrix.append(self.target_step - np.array(steps))
            elif v == 'last_loss':
                features_matrix.append(losses_arr)
            elif v == 'derivative_3':
                features_matrix.append(self._derivative_batch(losses_arr, nth_loss=3))
            else:
                raise ValueError(f"Unknown variable: {v}")
        
        X = np.column_stack(features_matrix) if features_matrix else np.empty((n, 0))
        self.losses = list(losses)  # sync state for any subsequent update() calls
        return self.equation(X)
    
    def _derivative_batch(self, losses: np.ndarray, nth_loss: int = 3) -> np.ndarray:
        """Vectorized derivative computation matching sequential behavior."""
        n = len(losses)
        result = np.zeros(n)
        for i in range(1, n):
            result[i] = derivative(losses[:i+1].tolist(), nth_loss=nth_loss)
        return result



