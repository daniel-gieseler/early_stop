import random
from functools import partial
from inspect import signature

# --- DECORATOR -------------------------------------------------------------

random.seed(0)
FEATURE_MARKET: dict[str, callable] = {}

def feature(
    cases: dict[str, list] = None,
    max_parameters: int = 2,
    max_cases: int = 4,
) -> callable:
    """Register a feature family. Usage: @feature({'case1': [v1_of_param_1, v1_of_param_2], 'case2': [v2_of_param_1, v2_of_param_2]})"""
    def _deco(fn):
        name = fn.__name__
        assert name not in FEATURE_MARKET, f"`{name}` is duplicated"
        arguments = list(signature(fn).parameters.keys())
        assert 'd' in arguments, f"`{name}` must have a 'd' argument"
        params = [a for a in arguments if a != 'd']
        assert len(arguments) <= max_parameters, f"`{name}` must have at most {max_parameters} arguments. Found: {arguments}"
        
        _cases = cases if cases is not None else {name: []}
        assert len(_cases) <= max_cases, f"`{name}` must have at most {max_cases} cases. Found: {_cases}"
        for case, values in _cases.items():
            variant_name = f'{name}_{case}' if cases is not None else name
            FEATURE_MARKET[variant_name] = partial(fn, **dict(zip(params, values)))
        print(f"registered {len(_cases)} cases of `{name}`")
        return fn
    return _deco


# --- FEATURES -----------------------------------------------------------

@feature()
def last_loss(d: dict) -> float:
    return d["loss"][-1]

@feature({
    'tail_10pct': [0.10],
    'tail_20pct': [0.20],
    'tail_30pct': [0.30],
})
def first_derivative(d: dict, *, tail_fraction: float) -> float:
    """Derivative of loss w.r.t. log-steps using linear regression on tail fraction."""
    steps = d["steps"]  # already in log-space
    losses = d["loss"]
    
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


@feature({
    'recent_half': [0.5],
    'recent_quarter': [0.25],
})
def loss_ratio(d: dict, *, reference_fraction: float) -> float:
    """Ratio of final loss to loss at an earlier reference point.
    
    In a power-law L(t) ~ t^(-alpha), this ratio directly encodes the
    effective decay rate. A smaller ratio indicates faster convergence.
    This is dimensionless and scale-invariant.
    """
    losses = d["loss"]
    n = len(losses)
    
    # Index for the reference point
    ref_idx = int(n * reference_fraction)
    ref_idx = max(0, min(ref_idx, n - 2))  # ensure valid index
    
    ref_loss = losses[ref_idx]
    final_loss = losses[-1]
    
    if ref_loss < 1e-10:  # avoid division by zero
        return 1.0
    
    return final_loss / ref_loss


@feature()
def aitken_extrapolated_limit(d: dict) -> float:
    """
    Aitken's Δ² projection of the asymptotic loss using the tail of the learning curve.
    Works directly on the last uniformly (log-step) spaced points. Falls back to last loss.
    """
    losses = d["loss"]
    n = len(losses)
    if n < 3:
        return losses[-1]

    tiny = 1e-12
    s_hat = None

    # Search from the tail for the last triple with nonzero second difference
    for i in range(n - 3, -1, -1):
        s0, s1, s2 = losses[i], losses[i + 1], losses[i + 2]
        d1 = s1 - s0
        d2 = s2 - 2 * s1 + s0
        if abs(d2) > tiny:
            s_hat = s0 - (d1 * d1) / d2
            break

    if s_hat is None:
        return losses[-1]

    # Never predict worse than the current best (monotone convergence)
    return min(losses[-1], s_hat)

# --- example ----------------------------------------------------------------

if __name__ == "__main__":
    from traditional_run import run_traditional_run
    TIMEOUT_IN_SECONDS = 60
    FEATURES = [
        'last_loss',
        'first_derivative_tail_10pct',
        'loss_ratio_recent_quarter',
        'aitken_extrapolated_limit'
    ]
    run_traditional_run(TIMEOUT_IN_SECONDS, FEATURES)