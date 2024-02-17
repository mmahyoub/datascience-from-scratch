'''
Module for implementing linear regression from scratch using native python and numpy package.
'''
import numpy as np 

class LinearRegression():
    def __init__(self, learning_rate:float, n_iterations:int):
        self.coefficients = None
        self.intercept = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X must be a numpy array."
        assert isinstance(y, np.ndarray), "y must be a numpy array."

        # Initialize coefficients and intercept (of linear equation B0 + B1 * x1 + B2 * x2 + ....)
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0.0

        # Run optimization algorithm to estimate the coefficients and intercept
        # Gradient Descent:
        for _ in range(self.n_iterations):
            y_pr = np.dot(X, self.coefficients) + self.intercept  #predicted ys given current values for coefficients and intercept

            ## Update coefficients and intercept ##
            # Gradients: dc and db, derivatives of cost function (mean squared error) with respect to coefficients and intercept resepctivley
            dc = (-2/n_samples) * np.dot(X.T, (y - y_pr)) 
            db = (-2/n_samples) * np.sum(y-y_pr)
            
            # Parameters update:
            self.coefficients = self.coefficients - self.learning_rate * dc
            self.intercept = self.intercept - self.learning_rate * db

    def predict(self, X):
        assert isinstance(X, np.ndarray), "X must be a numpy array."
        predictions = np.dot(X, self.coefficients) + self.intercept
        return predictions