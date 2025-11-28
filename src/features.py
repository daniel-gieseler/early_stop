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
    'tail_05pct': [0.05],
    'tail_10pct': [0.10],
    'tail_20pct': [0.20],
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


@feature({
    'tail_10pct': [0.10],
    'tail_20pct': [0.20],
})
def power_law_exponent(d: dict, *, tail_fraction: float) -> float:
    """Slope in log-log space: d(log_loss)/d(log_step).
    
    For power-law convergence L ~ step^(-alpha), this estimates -alpha.
    More negative values indicate faster power-law decay.
    Complements first_derivative which uses linear-log space.
    """
    import math
    
    steps = d["steps"]  # already in log-space
    losses = d["loss"]
    
    n = len(steps)
    window_size = max(2, int(n * tail_fraction))
    
    x = steps[-window_size:]
    y = [math.log(max(L, 1e-10)) for L in losses[-window_size:]]
    
    # Linear regression: slope = cov(x,y) / var(x)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    var = sum((xi - x_mean) ** 2 for xi in x)
    
    if var < 1e-10:
        return 0.0
    
    return cov / var

@feature({
    'tail_10pct': [0.10],
    'tail_20pct': [0.20],
    'tail_30pct': [0.30],
})
def second_derivative(d: dict, *, tail_fraction: float) -> float:
    """Second derivative of loss w.r.t. log-steps using quadratic regression on tail fraction."""
    steps = d["steps"]  # already in log-space
    losses = d["loss"]
    
    n = len(steps)
    window_size = max(3, int(n * tail_fraction))  # need at least 3 points for quadratic
    
    x = steps[-window_size:]
    y = losses[-window_size:]
    
    # Quadratic regression: y = a + b*x + c*x²
    # Second derivative = 2*c
    n_pts = len(x)
    
    # Compute sums for normal equations
    sx = sum(x)
    sy = sum(y)
    sx2 = sum(xi**2 for xi in x)
    sx3 = sum(xi**3 for xi in x)
    sx4 = sum(xi**4 for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sx2y = sum(xi**2 * yi for xi, yi in zip(x, y))
    
    # Solve normal equations for quadratic coefficient c
    # Using Cramer's rule for the c coefficient
    det = n_pts * (sx2 * sx4 - sx3**2) - sx * (sx * sx4 - sx2 * sx3) + sx2 * (sx * sx3 - sx2**2)
    
    if abs(det) < 1e-10:
        return 0.0
    
    det_c = n_pts * (sx * sx2y - sx2 * sxy) - sx * (sx * sx2y - sx2 * sxy) + sy * (sx * sx3 - sx2**2)
    c = det_c / det
    
    return 2 * c  # second derivative


@feature({
    'tail_10pct': [0.10],
    'tail_20pct': [0.20],
})
def tail_power_law_exponent(d: dict, *, tail_fraction: float) -> float:
    """
    Estimate the effective power-law exponent α in the converging tail.

    Assumes near the end: L(step) ≈ L_inf + C * step^{-α}.
    Since steps are already in log-space, we regress:
        log(L - L_inf) ~ (-α) * log_step + const
    Implementation details:
      - L_inf is estimated once via a robust Aitken Δ² pass on the tail.
      - We then fit a simple OLS slope on the last `tail_fraction` of points,
        after transforming to log(L - L_inf). Returns max(0, α̂).
    """
    from math import log

    steps = d["steps"]        # already log(step)
    losses = d["loss"]        # cumulative minimum; monotone non-increasing
    n = len(losses)
    if n < 3:
        return 0.0

    tiny = 1e-12

    # --- Aitken Δ² limit estimate (one pass from the tail); fallback to last loss ---
    s_hat = None
    for i in range(n - 3, -1, -1):
        s0, s1, s2 = losses[i], losses[i + 1], losses[i + 2]
        d1 = s1 - s0
        d2 = s2 - 2 * s1 + s0
        if abs(d2) > tiny:
            s_hat = s0 - (d1 * d1) / d2
            break
    if s_hat is None:
        s_hat = losses[-1]

    # Never predict worse than current best (monotone convergence)
    L_inf = min(losses[-1], s_hat)

    # --- Tail window selection ---
    window = max(2, int(n * tail_fraction))
    x_tail = steps[-window:]
    y_tail = losses[-window:]

    # --- Build log-gap series; skip non-positive gaps ---
    X, Y = [], []
    for xi, li in zip(x_tail, y_tail):
        gap = li - L_inf
        if gap > tiny:
            X.append(xi)
            Y.append(log(gap))

    if len(X) < 2:
        return 0.0

    # --- OLS slope of log-gap vs log-step: slope = -α ---
    x_mean = sum(X) / len(X)
    y_mean = sum(Y) / len(Y)
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(X, Y))
    var = sum((xi - x_mean) ** 2 for xi in X)
    if var < tiny:
        return 0.0

    slope = cov / var
    alpha_hat = -slope
    return alpha_hat if alpha_hat > 0.0 else 0.0



# --- example ----------------------------------------------------------------

if __name__ == "__main__":
    from traditional_run import run_traditional_run
    TIMEOUT_IN_SECONDS = 60
    FEATURES = [
        'last_loss',
        'first_derivative_tail_10pct',
    ]
    run_traditional_run(TIMEOUT_IN_SECONDS, FEATURES)