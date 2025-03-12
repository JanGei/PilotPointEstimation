import numpy as np

def smooth_step(x, step_positions, heights, alpha=10):
    """Creates a smooth step-like function using sigmoids.
    
    Parameters:
    x (array): Input x values.
    step_positions (array): Positions where steps occur.
    heights (array): Heights of the steps.
    alpha (float): Controls the smoothness of steps (higher = sharper).
    
    Returns:
    y (array): Step-like function values.
    """
    y = np.zeros_like(x)
    for x0, h in zip(step_positions, heights):
        y += h * (1 / (1 + np.exp(-alpha * (x - x0))))
    return y