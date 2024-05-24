import numpy as np
import sympy as sp

def newton_method_multi_dim(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for finding a local minimum of a function in multiple dimensions.

    Parameters:
    f (function): The objective function to minimize.
    grad_f (function): The gradient of the objective function.
    hess_f (function): The Hessian (matrix of second derivatives) of the objective function.
    x0 (np.array): Initial guess for the minimum.
    tol (float): Tolerance for stopping criteria.
    max_iter (int): Maximum number of iterations.

    Returns:
    np.array: Estimated position of the local minimum.
    """
    x = x0
    for i in range(max_iter+1):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Ensure the Hessian is non-singular
        if np.linalg.det(hess) == 0:
            raise ValueError("Hessian is singular")
        
        # Newton's update step
        hess_inv = np.linalg.inv(hess)
        x_new = x - hess_inv @ grad
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        
        x = x_new
    
    raise ValueError("Newton's method did not converge")

# Define the variables and function using SymPy
x1, x2 = sp.symbols('x1 x2')
f_sym = x1-2*x2+1.6*x1*x2+x1**2+x2**2

# Compute the gradient and Hessian using SymPy
grad_f_sym = [sp.diff(f_sym, var) for var in (x1, x2)]
hess_f_sym = [[sp.diff(g, var) for var in (x1, x2)] for g in grad_f_sym]

# Convert the symbolic expressions to numerical functions
grad_f_func = sp.lambdify((x1, x2), grad_f_sym, 'numpy')
hess_f_func = sp.lambdify((x1, x2), hess_f_sym, 'numpy')

# Wrapper functions to match the newton_method_multi_dim interface
def example_grad_f(x):
    return np.array(grad_f_func(x[0], x[1]))

def example_hess_f(x):
    return np.array(hess_f_func(x[0], x[1]))

# Initial guess
x0 = np.array([0.0, 0.0])

# Run Newton's method
minimum = newton_method_multi_dim(lambda x: f_sym.subs({x1: x[0], x2: x[1]}), example_grad_f, example_hess_f, x0)
print("The minimum is at:", minimum)
ans  = f_sym.subs({x1: minimum[0], x2: minimum[1]})
print(ans)
