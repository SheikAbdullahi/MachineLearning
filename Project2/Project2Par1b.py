import numpy as np
import matplotlib.pyplot as plt

# Define the function to create our dataset
def generate_data(n_samples=100):
    np.random.seed(42)  # for reproducibility
    X = 2 * np.random.rand(n_samples, 1)
    y = 1 + 2 * X + 3 * X**2 + np.random.randn(n_samples, 1)
    return X, y

# Initialize the dataset
X, y = generate_data()

# Add the bias (x0 = 1) and higher degree terms for polynomial regression
X_b = np.c_[np.ones((len(X), 1)), X, X**2]

# The cost function for a quadratic regression
def cost_function(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions - y))
    return cost

# Gradient of the cost function
def gradient(theta, X, y):
    m = len(y)
    return X.T.dot(X.dot(theta) - y) / m

# Gradient Descent with Momentum
def gradient_descent_with_momentum(X, y, theta, learning_rate, iterations, beta):
    m = len(y)
    v = np.zeros(theta.shape)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        gradients = gradient(theta, X, y)
        v = beta * v + (1 - beta) * gradients
        theta = theta - learning_rate * v
        cost_history[i] = cost_function(theta, X, y)

    return theta, cost_history

# Parameters
learning_rate = 0.05
iterations = 1000
beta = 0.9  # Momentum hyperparameter
initial_theta = np.random.randn(3, 1)

# Run gradient descent with momentum
theta, cost_history = gradient_descent_with_momentum(X_b, y, initial_theta, learning_rate, iterations, beta)

# Plotting the convergence
plt.plot(cost_history)
plt.title('Cost Function over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Final parameters
print("Final theta parameters:", theta.ravel())
