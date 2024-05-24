import numpy as np
from scipy.optimize import minimize_scalar
import sympy as sp
# Define the symbolic variables
x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')

# Define the objective function
def f(x):
    return 1*x[0] -1*x[1] + x[0]*x[1] + x[0]**2 + 2*x[1]**2

# Define the gradient of the objective function
def grad_f(x):
    x1_val, x2_val = x[0], x[1]
    grad_x1 = sp.diff(f([x1, x2]), x1).subs({x1: x1_val, x2: x2_val})
    grad_x2 = sp.diff(f([x1, x2]), x2).subs({x1: x1_val, x2: x2_val})
    return np.array([grad_x1, grad_x2]).astype(float)

def steepest_descent(f, grad_f, x0, tol=1e-6, max_iter=1):
    x = np.array(x0)
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        def line_search(lambda_val):
            return f(x - lambda_val * grad)
        # Perform line search to find optimal step size
        lambda_star = minimize_scalar(line_search).x
        x = x - lambda_star * grad
    return x

# Initial point
x0 = [0.39, -0.68]

# Run steepest descent algorithm
solution = steepest_descent(f, grad_f, x0)
func_val = f(solution)
print("Optimal solution:", solution)
print("f(x):", func_val)
