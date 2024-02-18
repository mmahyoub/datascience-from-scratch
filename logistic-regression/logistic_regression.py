'''
Module for implementing logistic regression for binary classification from scratch using native python and numpy. 

. Binary classification 
. Estimate probability of classes given a set of predictors
. Gradient descent for estimating the parameters of the model 
'''
import numpy as np 

class LogisticRegression():
    def __init__(self, learning_rate:float, n_iterations:int):
        self.coefficients = None
        self.intercept = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X must be a numpy array."
        assert isinstance(y, np.ndarray), "y must be a numpy array."

        # Initialize coefficients and intercept (of linear equation z = B0 + B1 * x1 + B2 * x2 + .... ==> sigmoid(-z))
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0.0

        # Run optimization algorithm to estimate the coefficients and intercept in sigmoid(-z); z = B0 + B1 * x1 + B2 * x2 + ....
        # Gradient Descent:
        for _ in range(self.n_iterations):
            z = np.dot(X, self.coefficients) + self.intercept  
            y_pr = self.sigmoid(z)  # estimate probabilities P(y | X)

            ## Update coefficients and intercept ##
            # Gradients: dc and db, derivatives of cost function (mean squared error) with respect to coefficients and intercept resepctivley
            dc = (1/n_samples) * np.dot(X.T, (y_pr-y)) 
            db = (1/n_samples) * np.sum(y_pr - y)
            
            # Parameters update:
            self.coefficients = self.coefficients - self.learning_rate * dc
            self.intercept = self.intercept - self.learning_rate * db

    def predict(self, X):
        assert isinstance(X, np.ndarray), "X must be a numpy array."

        z = np.dot(X, self.coefficients) + self.intercept
        predictions = [1 if p > 0.5 else 0 for p in self.sigmoid(z)]
        return predictions
    
    def predict_proba(self, X):
        assert isinstance(X, np.ndarray), 'X must be a numpy array.'

        z = np.dot(X, self.coefficients) + self.intercept
        p_class1 = self.sigmoid(z)
        p_class0 = 1 - p_class1
        probabilities = np.stack((p_class0, p_class1), axis = 1)
        return probabilities