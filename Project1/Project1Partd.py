import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

def create_design_matrix(x, y, degree):
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, l))

    index = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, index] = (x ** (i - j) * y ** j).flatten()
            index += 1

    return X

n_observations = 100
degree_max = 10
n_bootstrap = 100

x = np.random.rand(n_observations)
y = np.random.rand(n_observations)
z = FrankeFunction(x, y) + np.random.randn(n_observations) * 0.1

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, random_state=42)

bias_squared = np.zeros(degree_max)
variance = np.zeros(degree_max)
error = np.zeros(degree_max)

for degree in range(degree_max):
    X_train = create_design_matrix(x_train, y_train, degree=degree)
    X_test = create_design_matrix(x_test, y_test, degree=degree)

    y_pred = np.empty((z_test.shape[0], n_bootstrap))

    for i in range(n_bootstrap):
        x_, y_, z_ = resample(X_train, y_train, z_train)
        beta = np.linalg.pinv(x_.T @ x_) @ x_.T @ z_
        y_pred[:, i] = X_test @ beta

    bias_squared[degree] = np.mean((z_test - np.mean(y_pred, axis=1)) ** 2)
    variance[degree] = np.mean(np.var(y_pred, axis=1))
    error[degree] = np.mean(np.mean((y_pred - z_test.reshape(-1, 1)) ** 2, axis=1))

plt.plot(range(degree_max), bias_squared, label='Bias^2')
plt.plot(range(degree_max), variance, label='Variance')
plt.plot(range(degree_max), error, label='Error')
plt.plot(range(degree_max), bias_squared + variance, 'k--', label='Bias^2 + Variance')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('MSE')
plt.legend()
plt.show()
