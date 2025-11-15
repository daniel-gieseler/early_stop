import random

def dummy(curve: list[float, float]) -> callable:
    """
    This is a dummy function.
    Params should be determined based on the incomplete `curve`.
    Functions should contain a concise docstring.
    """
    param_a, param_b = random.random(), random.random()
    def loss(step: float) -> float:
        return step * param_b + param_a
    return loss