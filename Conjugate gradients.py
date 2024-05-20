import numpy as np
from scipy.optimize import minimize
import sympy as sp

# Define symbolic variables
x1, x2 = sp.symbols('x1 x2')

# Define the function using sympy
def f(x):
    return x[0] - x[1] + (-2.2) * x[0] * x[1] + 2 * x[0]**2 + x[1]**2

# Define the gradient of the function using sympy
def grad_f(x):
    df_dx1 = sp.diff(f([x1, x2]), x1).subs({x1: x[0], x2: x[1]})
    df_dx2 = sp.diff(f([x1, x2]), x2).subs({x1: x[0], x2: x[1]})
    return np.array([df_dx1, df_dx2], dtype=float)

# Define the starting point
x0 = np.array([0, 0])

# Perform optimization using conjugate gradients
result = minimize(f, x0, method='Newton-CG', jac=grad_f, options={'disp': True, 'maxiter': 2})

