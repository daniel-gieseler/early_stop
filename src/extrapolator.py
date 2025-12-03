from typing import Callable
import numpy as np

def derivative(loss: list, nth_loss: int = 1) -> float:
    last, found_idx = loss[-1], None
    for i in range(len(loss)-2, -1, -1):
        if loss[i] != last:
            found_idx = i
            nth_loss -= 1
            if nth_loss == 0:
                break
    return 0 if found_idx is None else (loss[found_idx] - last) / (found_idx - len(loss) + 1)

def derivative_batch(losses: list, nth_loss: int = 3) -> np.ndarray:
    return np.array([0] + [derivative(losses[:i+1], nth_loss) for i in range(1, len(losses))])

class Extrapolator:
    def __init__(self,
        target_step: int,
        variable_names: list[str] = ['delta_steps', 'last_loss', 'derivative_3'],
        equation: Callable[[np.ndarray], np.ndarray] = lambda x: x[:, 1] + x[:, 0] * (x[:, 2]*(x[:, 1]-0.41)-4.05e-6),
    ):
        self.equation = equation
        self.variable_names = variable_names
        self.target_step = target_step
        self.losses: list[float] = []
        
        self._feature_fns = {
            'delta_steps': (lambda step, _: self.target_step - step,
                            lambda steps, _: self.target_step - np.array(steps)),
            'last_loss': (lambda _, losses: losses[-1],
                          lambda _, losses: np.array(losses)),
            'derivative_3': (lambda _, losses: derivative(losses, nth_loss=3),
                             lambda _, losses: derivative_batch(losses, nth_loss=3)),
        }
        assert all(v in self._feature_fns for v in variable_names), f"Unknown variable in {variable_names}"

    def update(self, loss: float) -> float:
        self.losses.append(loss)
        features = [self._feature_fns[v][0](len(self.losses), self.losses) for v in self.variable_names]
        return self.equation(np.array([features])).item()

    def update_batch(self, losses: list[float]) -> np.ndarray:
        steps = list(range(1, len(losses) + 1))
        features = [self._feature_fns[v][1](steps, losses) for v in self.variable_names]
        return self.equation(np.column_stack(features))