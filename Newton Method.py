import sympy as sp

# Define the variable and function
x = sp.symbols('x')
f = 0.57*x**3 + 4.97*x**2 + 1.79*x + 1.67

# Compute the first and second derivatives
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f_prime, x)

# Convert the SymPy expressions to numerical functions
f_prime_func = sp.lambdify(x, f_prime, 'numpy')
f_double_prime_func = sp.lambdify(x, f_double_prime, 'numpy')

# Newton's method implementation
def newton_method(f_prime, f_double_prime, x0, tolerance=1e-10, max_iterations=1000):
    x = x0
    for i in range(max_iterations):
        f_prime_x = f_prime(x)
        f_double_prime_x = f_double_prime(x)
        if f_double_prime_x == 0:
            print("Zero second derivative. No solution found.")
            return None
        x_new = x - f_prime_x / f_double_prime_x
        if abs(x_new - x) < tolerance:
            return x_new
        x = x_new
    print("Exceeded maximum iterations. No solution found.")
    return None

# Initial guess
x0 = 0  # A reasonable initial guess (can be adjusted if necessary)

# Find the minimum
minimum_x = newton_method(f_prime_func, f_double_prime_func, x0)
print(f"The value of x at the minimum is {minimum_x:.4f}")
