from pysr import PySRRegressor

def fit_pysr_model_on_df_customized(df, epsilon = 0.004, timeout_in_seconds = 60):
    X = df[["delta_steps", "last_loss", "derivative_3", "curve_id"]]
    y = df["target_loss"]

    delta_max = float(X["delta_steps"].max())

    loss_function = f"""
    function curve_safe_delta_loss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
        X = dataset.X
        y = dataset.y
        pred, complete = eval_tree_array(tree, X, options)
        if !complete
            return L(Inf)
        end
        n = length(pred)
        deltas    = X[1, :]   # delta_steps
        curve_ids = round.(Int, X[4, :])  # curve_id (4th column in X)

        eps     = L({epsilon})
        Δ_max_L = L({delta_max})

        max_safe = Dict{{Int, L}}()

        @inbounds for i in 1:n
            g = curve_ids[i]
            if abs(pred[i] - y[i]) <= eps
                d = L(deltas[i])
                if haskey(max_safe, g)
                    if d > max_safe[g]
                        max_safe[g] = d
                    end
                else
                    max_safe[g] = d
                end
            end
        end

        groups_seen = unique(curve_ids)
        total_penalty = zero(L)

        for g in groups_seen
            if haskey(max_safe, g)
                safe_d = max_safe[g]
                total_penalty += (Δ_max_L - safe_d)
            else
                total_penalty += Δ_max_L
            end
        end

        return total_penalty / L(length(groups_seen))
    end
    """

    model = PySRRegressor(
        maxsize=20,
        niterations=10_000_000,
        timeout_in_seconds=timeout_in_seconds,
        maxdepth=16,
        binary_operators=["+", "*", "-", "/", "pow"],
        unary_operators=[
            "exp",
            "log",
            "sqrt",
        ],
        precision=16,
        constraints={
            "pow": (-1, 4),
            "sqrt": (1),
        },
        nested_constraints={
            "log": {"log": 1, "pow": 1, "exp": 1},
            "pow": {"log": 1, "pow": 1, "exp": 1},
            "exp": {"log": 1, "pow": 1, "exp": 1},
        },
        loss_function=loss_function,
        complexity_of_variables=[1.0, 1.0, 1.0, 1e6],
        batching=False,
    )

    model.fit(X, y)

    return model