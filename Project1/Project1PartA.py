# Generate Data Points (x, y) and compute Franke Function.
# Add Stochastic Noise if needed.
# Create Design Matrix for up to 5th order polynomial.
# Split the data into training and test sets.
# Scale/Center the Data.
# Implement OLS and find β-coefficients.
# Evaluate models using MSE and R^2 score.
# Plot the results





import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate Data Points (x, y) and compute Franke Function.
np.random.seed(42)  # for reproducibility
N = 1000  # number of data points
x = np.random.rand(N)
y = np.random.rand(N)

def FrankeFunction(x,y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)
noise_strength = 0.1
z = z + noise_strength * np.random.randn(N)  # Step 2: Adding Stochastic Noise

# Step 3: Create Design Matrix
def create_design_matrix(x, y, degree=5):
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    index = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, index] = ((x ** (i - j)) * (y ** j)).flatten()  # Flatten the array before assigning
            index += 1

    return X



X = create_design_matrix(x, y)

# Step 4: Split the data into training and test sets
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

# Step 5: Scale/Center the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Implement OLS and find β-coefficients.
beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train

# Step 7: Evaluate models using MSE and R^2 score
z_pred = X_test_scaled @ beta
mse = mean_squared_error(z_test, z_pred)
r2 = r2_score(z_test, z_pred)

print(f"MSE: {mse}, R^2: {r2}")

# Step 8: Plot the results
plt.figure(figsize=(10,6))
plt.scatter(z_test, z_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('OLS Prediction of Franke Function')
plt.grid(True)
plt.show()
