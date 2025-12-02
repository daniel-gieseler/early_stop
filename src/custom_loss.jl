function evaluate_early_stop(X, y, y_hat; epsilon = 0.001, target_step = 4300)
    """
    Assumes X to be an array of size (n, 4) with columns 
    ['delta_steps', 'last_loss', 'derivative_3', 'curve_id'].
    Assumes X, y and y_hat to be the same length.
    """
    early_stops_aux = Dict{Int, Float64}()

    # Julia is 1-based, so we loop from 1 to size(X, 1)
    for i in 1:size(X, 1)
        if abs(y_hat[i] - y[i]) <= epsilon
            delta_steps = X[i, 1]
            curve_id = Int(X[i, 4])

            # get(dictionary, key, default) in Julia
            early_stops_aux[curve_id] = max(get(early_stops_aux, curve_id, 0.0), delta_steps)
        end
    end

    delta_step_avg = sum(values(early_stops_aux)) / length(early_stops_aux)
    return target_step - delta_step_avg
end