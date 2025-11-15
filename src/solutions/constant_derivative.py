def constant_derivative(curve: list[tuple[float, float]]) -> callable:
    """
    Fits a constant first derivative model to the curve.
    Computes average derivative from the curve data.
    """
    if len(curve) < 2:
        # Fallback for very short curves
        def loss(step: float) -> float:
            return -0.001
        return loss
    
    # Compute first derivatives from the curve
    derivatives = []
    for i in range(len(curve) - 1):
        step1, loss1 = curve[i]
        step2, loss2 = curve[i + 1]
        if step2 != step1:
            derivative = (loss2 - loss1) / (step2 - step1)
            derivatives.append(derivative)
    
    # Use average derivative (or recent average for better prediction)
    avg_derivative = sum(derivatives) / len(derivatives) if derivatives else -0.001
    
    def loss(step: float) -> float:
        return avg_derivative
    
    return loss

