import sympy as sp
from scipy.optimize import differential_evolution
import numpy as np

# Define the variable
alpha = sp.symbols('alpha')

# Define the lift coefficient C_L
C_L = 0.09 * (alpha + 2)

# Define the drag coefficient C_D
C_D = 0.02 + 0.055 * C_L**2

# Define the lift-to-drag ratio
L_D_ratio = C_L / C_D

def objective_f(x):
    # Ensure the function can handle both scalar and array inputs
    if isinstance(x, (list, np.ndarray)):
        return np.array([-float(L_D_ratio.subs({alpha: val})) for val in x])
    else:
        return -float(L_D_ratio.subs({alpha: x}))

lower_bound = -100
upper_bound = 100

result = differential_evolution(objective_f, bounds=[(lower_bound, upper_bound)])
optimal_x = result.x[0]
best_objective = -objective_f(optimal_x)

# Print the best lift-to-drag ratio to six decimal places
print(f"Best angle: {optimal_x:.4f}")
print(f"Best lift-to-drag ratio: {best_objective:.4f}")
