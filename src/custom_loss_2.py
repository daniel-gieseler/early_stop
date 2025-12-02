from pysr import PySRRegressor

def fit_pysr_model_on_df_customized(
    df,
    epsilon: list[float] = [0.005, 0.01, 0.02, 0.04],
    timeout_in_seconds: int = 60,
):
    X = df[["delta_steps", "last_loss", "derivative_3"]]#, "curve_id"]]
    y = df["target_loss"]

    # Julia loss function, with evaluate_early_stop defined and used:
    epsilons_str = "[" + ", ".join(map(str, epsilon)) + "]"
    loss_function = f"""
    # X is features × samples:
    # 1 -> delta_steps
    # 2 -> last_loss
    # 3 -> derivative_3
    # 4 -> curve_id
    function evaluate_early_stop(X, y, y_hat; epsilons={epsilons_str})
        early_stops_aux = Dict{{Int, Dict{{Float64, Float64}}}}()
        n_samples = size(X, 2)

        for epsilon in epsilons
            for j in 1:n_samples
                if abs(y_hat[j] - y[j]) <= epsilon
                    delta_steps = X[1, j]      # delta_steps column
                    curve_id    = Int(X[4, j]) # curve_id column

                    if !haskey(early_stops_aux, curve_id)
                        early_stops_aux[curve_id] = Dict{{Float64, Float64}}()
                    end

                    prev = get(early_stops_aux[curve_id], epsilon, 0.0)
                    early_stops_aux[curve_id][epsilon] = max(prev, delta_steps)
                end
            end
        end

        # If no points are within any epsilon, return 0
        if isempty(early_stops_aux)
            return 0.0
        end

        # Average over all curve_id and epsilon combinations
        total_delta = 0.0
        count = 0
        for curve_dict in values(early_stops_aux)
            for delta_step in values(curve_dict)
                total_delta += delta_step
                count += 1
            end
        end

        delta_step_avg = total_delta / count
        return delta_step_avg
    end

    function eval_loss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end

        # es = evaluate_early_stop(
        #     dataset.X,
        #     dataset.y,
        #     prediction;
        #     epsilons={epsilons_str},
        # )

        mse = sum((prediction .- dataset.y) .^ 2) / dataset.n
        # mse_squared = mse * mse

        #es_log = log(max(L(es), 1.1))
        #return L(mse)/es_log
        return L(mse)
    end
    """

    model = PySRRegressor(
        maxsize=22,
        niterations=10_000_000,
        timeout_in_seconds=timeout_in_seconds,
        maxdepth=16,
        binary_operators=["+", "*", "-", "/", "pow"],
        unary_operators=["exp", "log"],
        precision=16,
        constraints={
            "pow": (-1, 4),
        },
        nested_constraints={
            "log": {"log": 0, "pow": 0, "exp": 0},
            "pow": {"log": 1, "pow": 0, "exp": 0},
            "exp": {"log": 1, "pow": 0, "exp": 0},
        },
        #loss_function=loss_function,
        #complexity_of_variables=[1.0, 1.0, 1.0, 1e6],
        batching=False,
    )

    model.fit(X, y)

    return model
