df = ...
X = df[["delta_steps", "last_loss", "derivative_3", "curve_id"]].to_numpy()

def evaluate_early_stop(X, y, y_hat, epsilon = 0.001, target_step = 4300):
    """
    Assumes X to be numpy array of shape (n, 4) with columns ['delta_steps', 'last_loss', 'derivative_3', 'curve_id'].
    Assumes X, y and y_hat to be same length n.
    """
    early_stops_aux = {}
    for i in range(len(X)):
        if abs(y_hat[i] - y[i]) <= epsilon:
            delta_steps, curve_id = X[i][0], int(X[i][3])
            early_stops_aux[curve_id] = max(early_stops_aux.get(curve_id, 0), delta_steps)
    delta_step_avg = sum(early_stops_aux.values()) / len(early_stops_aux)
    return target_step - delta_step_avg


