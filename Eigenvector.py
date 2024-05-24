import numpy as np

def compute_priority_vector(matrix):
    """
    Computes the priority vector (eigenvector) for the given preference matrix.
    
    Parameters:
    matrix (numpy.ndarray): The matrix representing relative preferences.

    Returns:
    numpy.ndarray: The normalized priority vector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    priority_vector = eigenvectors[:, max_eigenvalue_index]
    # Normalize the eigenvector
    priority_vector = priority_vector / np.sum(priority_vector)
    # Return the real part of the priority vector (in case of complex numbers)
    return np.real(priority_vector), eigenvalues

# Define the relative importance
# p13 = p12 * p23
p12 = 2.8
p23 = 4.36
p13 = p12 * p23

# Construct the preference matrix
preference_matrix = np.array([
    [1, p12, p13],
    [1/p12, 1, p23],
    [1/p13, 1/p23, 1]
])

# Compute the priority vector
weight_vector = compute_priority_vector(preference_matrix)[0]
eigen_values = compute_priority_vector(preference_matrix)[1]

# Print the priority vector
print("Weight Vector:", weight_vector)