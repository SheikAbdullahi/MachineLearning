import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data for testing
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Function to compute the gradient of the cost function
def compute_gradient(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) / len(X_b)

# Function to calculate the cost
def compute_cost(theta, X_b, y):
    return np.sum((X_b.dot(theta) - y)**2) / (2 * len(X_b))

# Mini-batch SGD function
def mini_batch_SGD(X, y, learning_rate=0.01, batch_size=20, num_epochs=50):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]  # Add x0 = 1 to each instance
    theta = np.random.randn(2, 1)  # Initialization

    cost_history = []  # For plotting

    for epoch in range(num_epochs):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, batch_size):
            xi = X_b_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = 2/batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients
            cost = compute_cost(theta, X_b, y)
            cost_history.append(cost)

    return theta, cost_history

# Parameters
learning_rate = 0.01
batch_size = 20
num_epochs = 50

# Run mini-batch SGD
theta, cost_history = mini_batch_SGD(X, y, learning_rate, batch_size, num_epochs)

# Plotting the cost history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()

print(f"The final theta values are: {theta.ravel()}")
