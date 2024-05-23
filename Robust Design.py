import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

def f(x):
    return x + 1 / x

def mean_performance_index(x, noise):
    lower_bound = x - noise
    upper_bound = x + noise
    mean_value, _ = quad(f, lower_bound, upper_bound)
    mean_value /= (upper_bound - lower_bound)  # Normalize by the interval length
    return mean_value

def mean_performance_index_to_minimize(x, noise):
    # Since minimize works with arrays, we need to handle the input accordingly
    return mean_performance_index(x[0], noise)

noise = 0.5
lower_bound = 0.75
upper_bound = 2
#initial_guess = [1.0]  # Initial guess for the optimizer

result_nominal_design = differential_evolution(f, bounds=[(lower_bound, upper_bound)])
optimal_x_nominal_design = result_nominal_design.x[0]
penalty_nominal = np.abs(mean_performance_index(optimal_x_nominal_design, noise)-f(optimal_x_nominal_design))/f(optimal_x_nominal_design)

result_mean_curve = differential_evolution(mean_performance_index_to_minimize, args=(noise,), bounds=[(lower_bound, upper_bound)])
optimal_x_mean_curve = result_mean_curve.x[0]
minimum_x_based_on_mean = f(optimal_x_mean_curve)
penalty_mean = np.abs(mean_performance_index(optimal_x_mean_curve, noise)-f(optimal_x_nominal_design))/f(optimal_x_nominal_design)

uncertainty_in_fx = 1-result_nominal_design.fun/minimum_x_based_on_mean


print(f"Optimal x for f(x): {optimal_x_nominal_design}")
print(f"Minimum f(x): {result_nominal_design.fun}")
print(f"Penalty for direct optimisation: {penalty_nominal}")
print("------------------------------------------------")
print(f"Optimal x for E[f(x)]: {optimal_x_mean_curve}")
print(f"f(x) based on optimal x for mean: {minimum_x_based_on_mean}")
print(f"Minimum E[f(x)]: {result_mean_curve.fun}")
print(f"Penalty for noise adjusted optimisation: {penalty_mean}")
print("------------------------------------------------")
print(f"Uncertainty in f(x): {uncertainty_in_fx}")