import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Define the FrankeFunction
def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Define create_design_matrix function here
def create_design_matrix(x, y, degree=5):
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

# Generate your dataset
n_observations = 1000
x = np.random.rand(n_observations)
y = np.random.rand(n_observations)
z = FrankeFunction(x, y) + np.random.randn(n_observations) * 0.1  # Added Noise

# Split data into training and testing sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, random_state=42)

# Initialize the k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store the Mean Squared Error for each model
ols_mse = []
ridge_mse = []
lasso_mse = []

degree_max = 10

for degree in range(degree_max):
    X_train = create_design_matrix(x_train, y_train, degree=degree)
    X_test = create_design_matrix(x_test, y_test, degree=degree)

    ols_mse_degree = []
    ridge_mse_degree = []
    lasso_mse_degree = []

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        z_train_fold, z_test_fold = z_train[train_index], z_train[test_index]

        ols = LinearRegression().fit(X_train_fold, z_train_fold)
        ols_mse_degree.append(mean_squared_error(z_test_fold, ols.predict(X_test_fold)))

        ridge = Ridge(alpha=0.1).fit(X_train_fold, z_train_fold)
        ridge_mse_degree.append(mean_squared_error(z_test_fold, ridge.predict(X_test_fold)))

        lasso = Lasso(alpha=0.1).fit(X_train_fold, z_train_fold)
        lasso_mse_degree.append(mean_squared_error(z_test_fold, lasso.predict(X_test_fold)))

    ols_mse.append(np.mean(ols_mse_degree))
    ridge_mse.append(np.mean(ridge_mse_degree))
    lasso_mse.append(np.mean(lasso_mse_degree))

plt.plot(range(degree_max), ols_mse, label='OLS MSE')
plt.plot(range(degree_max), ridge_mse, label='Ridge MSE')
plt.plot(range(degree_max), lasso_mse, label='Lasso MSE')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Compare the obtained MSEs with the ones obtained from the Bootstrap method.
