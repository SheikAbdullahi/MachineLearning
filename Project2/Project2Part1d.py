import numpy as np

# Adagrad implementation
def adagrad(X, y, num_iterations, learning_rate, epsilon, use_momentum=False, momentum_beta=0.9):
    num_features = X.shape[1]
    theta = np.zeros(num_features)
    accum_grad = np.zeros(num_features)  # Accumulated gradient
    v = np.zeros(num_features)  # Momentum
    
    for t in range(1, num_iterations + 1):
        gradient = compute_gradient(X, y, theta) 
        accum_grad += gradient**2
        adjusted_lr = learning_rate / (np.sqrt(accum_grad) + epsilon)
        
        if use_momentum:
            v = momentum_beta * v + adjusted_lr * gradient
            theta -= v
        else:
            theta -= adjusted_lr * gradient

    return theta

# SGD with Adagrad
def sgd_adagrad(X, y, learning_rate, epsilon, batch_size, num_epochs, use_momentum=False, momentum_beta=0.9):
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)
    accum_grad = np.zeros(num_features)  # Accumulated gradient
    v = np.zeros(num_features)  # Momentum

    for epoch in range(num_epochs):
        for i in range(0, num_samples, batch_size):
            end = i + batch_size
            X_batch = X[i:end]
            y_batch = y[i:end]

            gradient = compute_gradient(X_batch, y_batch, theta)  # Implement compute_gradient
            accum_grad += gradient**2
            adjusted_lr = learning_rate / (np.sqrt(accum_grad) + epsilon)

            if use_momentum:
                v = momentum_beta * v + adjusted_lr * gradient
                theta -= v
            else:
                theta -= adjusted_lr * gradient
            
    return theta

# Plain GD with Adagrad
theta_plain_adagrad = adagrad(X, y, num_iterations=1000, learning_rate=0.1, epsilon=1e-8)

# Plain GD with Adagrad and Momentum
theta_plain_adagrad_momentum = adagrad(X, y, num_iterations=1000, learning_rate=0.1, epsilon=1e-8, use_momentum=True)

# SGD with Adagrad
theta_sgd_adagrad = sgd_adagrad(X, y, learning_rate=0.1, epsilon=1e-8, batch_size=32, num_epochs=100)

# SGD with Adagrad and Momentum
theta_sgd_adagrad_momentum = sgd_adagrad(X, y, learning_rate=0.1, epsilon=1e-8, batch_size=32, num_epochs=100, use_momentum=True)
