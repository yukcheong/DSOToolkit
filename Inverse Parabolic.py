def f(x):
    return x**5 - 2*x**3 - 10*x + 5

def inverse_parabolic_interpolation(f, x1, x2, x3):
    f1, f2, f3 = f(x1), f(x2), f(x3)
    numerator = (f3 - f2)*(x2**2-x1**2)+(f1-f2)*(x3**2-x2**2)
    denominator = (f3 - f2)*(x2-x1)+(f1-f2)*(x3-x2)
    
    if denominator == 0:
        raise ValueError("Denominator in inverse parabolic interpolation became zero.")
    
    x_min = 0.5 * numerator / denominator
    return x_min

def minimize_using_inverse_parabolic_interpolation(f, x1, x2, x3, iterations):
    points = [(x1, f(x1)), (x2, f(x2)), (x3, f(x3))]
    print(f"Initial points: {points}")
    
    for i in range(iterations):
        x_new = inverse_parabolic_interpolation(f, x1, x2, x3)
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {f(x_new)}")
        
        # Update points: keep the best two old points and add the new one
        points.append((x_new, f(x_new)))
        points.sort(key=lambda p: p[1])  # Sort points by function values
        points = points[:3]  # Keep only the best three points
        
        x1, x2, x3 = points[0][0], points[1][0], points[2][0]
    
    return x_new

x1, x2, x3 = 0, 1, 2
iterations = 2
x_min = minimize_using_inverse_parabolic_interpolation(f, x1, x2, x3, iterations)
print(f"Minimum found: x = {x_min}, f(x) = {f(x_min)}")