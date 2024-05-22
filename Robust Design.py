from scipy.integrate import quad
from scipy.optimize import minimize

def f(x):
    return x + 1 / x

def mean_performance_index(x, noise):
    lower_bound = x - noise
    upper_bound = x + noise
    mean_value, _ = quad(f, lower_bound, upper_bound)
    return mean_value

def mean_performance_index_to_minimize(x, noise):
    # Since minimize works with arrays, we need to handle the input accordingly
    return mean_performance_index(x[0], noise)

noise = 0.75
lower_bound = 0.8
upper_bound = 2
initial_guess = [1.0]  # Initial guess for the optimizer

result_mean_curve = minimize(mean_performance_index_to_minimize, initial_guess, args=(noise,), bounds=[(lower_bound, upper_bound)])
optimal_x_mean_curve = result_mean_curve.x[0]
minimum_x_based_on_mean = f(optimal_x_mean_curve)

result_nominal_design = minimize(f, initial_guess, bounds=[(lower_bound, upper_bound)])
optimal_x_nominal_design = result_nominal_design.x[0]

print(f"Optimal x for f(x): {optimal_x_nominal_design}")
print(f"Minimum f(x): {result_nominal_design.fun}")
print("------------------------------------------------")
print(f"Optimal x for E[f(x)]: {optimal_x_mean_curve}")
print(f"Minimum f(x) based on mean: {minimum_x_based_on_mean}")
print(f"Minimum E[f(x)]: {result_mean_curve.fun}")