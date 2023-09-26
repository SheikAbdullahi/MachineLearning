import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
# Python implementation to perform Lasso Regression on the Franke function
# Assuming FrankeFunction and create_design_matrix are already correctly implemented
def FrankeFunction(x, y):
    # Your actual implementation here
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

def create_design_matrix(x, y, degree=5):
    # Your actual implementation here
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(y.shape) == 1:
        y = y[:, np.newaxis]
        
    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)  # Number of terms in the polynomial
    X = np.ones((N, l))
    
    index = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, index] = (x ** (i - j) * y ** j).flatten()
            index += 1
            
    return X

# Generate your x, y here
np.random.seed(42)
N = 1000
x = np.random.rand(N)
y = np.random.rand(N)
z = FrankeFunction(x, y)

# Create Design Matrix
X = create_design_matrix(x, y, degree=5)

# Split the data into training and test sets
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

# Scale/Center the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up the lambda values
lambdas = np.logspace(-4, 5, 10)

mse_scores = []
r2_scores = []

for lambd in lambdas:
    # Perform Lasso Regression
    lasso_reg = Lasso(alpha=lambd, max_iter=10000)
    lasso_reg.fit(X_train_scaled, z_train)
    
    # Make predictions
    z_pred_lasso = lasso_reg.predict(X_test_scaled)
    
    # Calculate and store the MSE and R^2 scores
    mse_scores.append(mean_squared_error(z_test, z_pred_lasso))
    r2_scores.append(r2_score(z_test, z_pred_lasso))

# Plotting
fig, ax1 = plt.subplots()

ax1.set_xscale('log')
ax1.set_xlabel('Lambda')
ax1.set_ylabel('MSE', color='tab:blue')
ax1.plot(lambdas, mse_scores, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  
ax2.set_ylabel('R^2', color='tab:red')  
ax2.plot(lambdas, r2_scores, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()  
plt.show()
