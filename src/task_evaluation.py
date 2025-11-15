
def extrapolate(
    curve: list[tuple[float, float]],
    cut_id: int,
    train_first_derivative: callable,
) -> list[tuple[float, float]]:
    first_derivative = train_first_derivative(curve[:cut_id+1])
    next_steps = [step for step, _ in curve[cut_id+1:]]
    extrapolation = [curve[cut_id]]
    for next_step in next_steps: 
        step, loss = extrapolation[-1]
        loss += first_derivative(next_step) * (next_step - step)
        extrapolation.append((next_step, loss))
    return extrapolation

def extrapolate_curves(
    curves: list[list[tuple[float, float]]],
    epsilon: float,
    train_first_derivative: callable,
):
    results = []
    for curve in curves:
        save = True
        minimum_loss = min(curve, key=lambda x: x[1])[1]
        results.append({
            'curve': curve,
            'minimum_loss': minimum_loss,
            'margin': abs(epsilon * minimum_loss),
            'predicted_loss': [],
        })
        for cut_id in reversed(range(len(curve) - 1)):
            extrapolated_loss = extrapolate(curve, cut_id, train_first_derivative)
            predicted_loss = extrapolated_loss[-1][1]
            error = abs((minimum_loss - predicted_loss) / minimum_loss)
            if error > epsilon:
                save = False
            if save:
                results[-1]['extrapolation'] = extrapolated_loss
                results[-1]['step'] = curve[cut_id][0]
            results[-1]['predicted_loss'].append((curve[cut_id][0], predicted_loss))
    return results
