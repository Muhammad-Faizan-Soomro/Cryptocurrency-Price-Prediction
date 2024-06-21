
# Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# X, y = Load Dataset Here @Faizan
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train,
y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# <<<<<----------------------------------->>>>>



# Linear Regression without python package with normal equation

import numpy as np

def linear_regression(X, y):
    X = np.column_stack((np.ones(len(X)), X))
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def predict(X, theta):
    X = np.column_stack((np.ones(len(X)), X))
    y_pred = X.dot(theta)
    return y_pred

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

# X = Load Dataset Here @Faizan
# y = Load Dataset Here @Faizan

# Train the model
theta = linear_regression(X, y)

# Make predictions
y_pred = predict(X, theta)

# Calculate mean squared error
mse = mean_squared_error(y, y_pred)

print("Model coefficients (theta):", theta.flatten())
print("Mean Squared Error:", mse)

# <<<<<----------------------------------->>>>>
