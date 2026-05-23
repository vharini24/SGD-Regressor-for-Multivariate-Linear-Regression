# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create a dataset with multiple input features and target values, then normalize the feature data and add a bias term.

2. Initialize model weights and set hyperparameters such as learning rate and number of epochs.

3. Train the model using Stochastic Gradient Descent by predicting output, calculating error, and updating weights for each training sample.

4. Use the trained weights to predict output values for the dataset and display the predicted results.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: V. HARINI
RegisterNumber:  212225040113
*/

import numpy as np

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
# Features: [Hours Studied, Attendance, Previous Marks]
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target: Marks Scored
y = np.array([50, 45, 70, 80, 95], dtype=float)

# ------------------------------
# Step 2: Feature normalization
# ------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

# ------------------------------
# Step 3: Initialize weights
# ------------------------------
n_features = X.shape[1]
weights = np.zeros(n_features)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# ------------------------------
# Step 4: Stochastic Gradient Descent
# ------------------------------
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)

```

## Output:

<img width="852" height="57" alt="image" src="https://github.com/user-attachments/assets/c5e85496-c620-41cc-a305-d11fb58ce73b" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
