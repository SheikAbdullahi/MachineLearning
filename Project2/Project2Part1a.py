import numpy as np

# Function to compute the cost
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

# Function to perform gradient descent
def gradient_descent(X, y, theta, learning_rate, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y.flatten()  # Ensuring errors is a 1D array
        delta = (1/m) * X.T.dot(errors)  # This should now be a shape (number of features,)
        theta -= learning_rate * delta
        cost_history[i] = compute_cost(X, y, theta)
        
        # Print the cost every 100 iterations (optional)
        if i % 100 == 0:
            print(f"Iteration {i+1}/{num_iters} | Cost: {cost_history[i]:.4f}")
    
    return theta, cost_history

# Example data generation
np.random.seed(42)  # Seed for reproducibility
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + x**2 + np.random.randn(100, 1)  # Quadratic equation a_0=4, a_1=3, a_2=1

# Prepare the data matrix with a column of ones for the bias term (a_0)
X_b = np.c_[np.ones((len(x), 1)), x, x**2]

# Parameters initialization
theta_initial = np.zeros(X_b.shape[1])

# Hyperparameters for the Gradient Descent
learning_rate = 0.1
num_iters = 1000

# Run Gradient Descent
theta_optimal, cost_history = gradient_descent(X_b, y, theta_initial, learning_rate, num_iters)

print('Optimal parameters:', theta_optimal)

# Predictions can be made with the optimal theta and new input data
# new_x = ... (new input data)
# new_X_b = np.c_[np.ones((len(new_x), 1)), new_x, new_x**2]
# predictions = new_X_b.dot(theta_optimal)

# Optionally, you can plot the cost history over iterations to see how it decreases
import matplotlib.pyplot as plt

plt.plot(range(1, num_iters+1), cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()
