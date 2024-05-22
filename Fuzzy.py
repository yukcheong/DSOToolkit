import numpy as np
import matplotlib.pyplot as plt

lower_bound = 0.5
upper_bound = 2

# Define the x and y values of your two points
x = np.array([lower_bound, upper_bound])
y = np.array([1, 0])

# Perform linear regression
coefficients = np.polyfit(x, y, 1)

# Define the functions f1 and f2
def f1(x):
    return 1 / x

def f2(x):
    return x ** 2

# Define the membership function
def membership(f):
    return np.where(f <= lower_bound, 1.0, np.where(f >= upper_bound, 0.0, coefficients[0]*f + coefficients[1]))

# Define the range of x
x_values = np.linspace(lower_bound, upper_bound, 100000000)  # avoid division by zero

# Compute the values of f1 and f2
f1_values = f1(x_values)
f2_values = f2(x_values)

# Compute the membership values
memb1_values = membership(f1_values)
memb2_values = membership(f2_values)

# Combine the membership values (e.g., using the addition operator)
combined_membership = memb1_values + memb2_values

# Find the index of the maximum combined membership value (optimal)
optimal_index = np.argmax(combined_membership)
optimal_x = x_values[optimal_index]
optimal_f1 = f1_values[optimal_index]
optimal_f2 = f2_values[optimal_index]
optimal_combined = membership(optimal_f1) + membership(optimal_f2)

# Find the index of the minimum combined membership value (worst)
worst_index = np.argmin(combined_membership)
worst_x = x_values[worst_index]
worst_f1 = f1_values[worst_index]
worst_f2 = f2_values[worst_index]
worst_combined = membership(worst_f1) + membership(worst_f2)

# Define the starting point
x0 = np.array([0, 0])

print(f"Optimal x: {optimal_x}")
print(f"Combined({optimal_x}) = {optimal_combined}")
print(f"f1({optimal_x}) = {optimal_f1}")
print(f"f2({optimal_x}) = {optimal_f2}")

print(f"Worst x: {worst_x}")
print(f"Combined({optimal_x}) = {worst_combined}")
print(f"f1({worst_x}) = {worst_f1}")
print(f"f2({worst_x}) = {worst_f2}")

# Plotting the membership functions and the combined membership

'''
plt.plot(x_values, memb1_values, label='memb1')
plt.plot(x_values, memb2_values, label='memb2')
plt.plot(x_values, combined_membership, label='combined', linestyle='--')
plt.axvline(x=optimal_x, color='green', linestyle=':', label=f'Optimal x = {optimal_x:.6f}')
plt.axvline(x=worst_x, color='red', linestyle=':', label=f'Worst x = {worst_x:.6f}')
plt.xlabel('Variable x')
plt.ylabel('Membership')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Membership Functions and Combined Membership')
plt.show()
'''

