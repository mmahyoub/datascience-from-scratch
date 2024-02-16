'''
Two Classes: 
    1. GaussianNB
    2. CategoricalNB
'''
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class GaussianNB:
    '''
    Build a naive bayes classifier from scratch using Python (native python and numpy). 

    Gaussian Naive Bayes.

    Assumptions:
    1. Features are normally distributed 
    2. Features are independent 
    '''
    def __init__(self):
        self.classes = None
        self.class_probabilities = None
        self.mean = None
        self.variance = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probabilities = np.zeros(len(self.classes))
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.variance = np.zeros((len(self.classes), X.shape[1]))

        for i, c in enumerate(self.classes):
            # Get rows corresponding to class c
            X_class = X[y==c]

            # Compute the class probability
            self.class_probabilities[i] = np.sum(y == c) / len(y)

            # Compute mean and variance 
            self.mean[i] = np.mean(X_class, axis = 0)
            self.variance[i] = np.var(X_class, axis = 0)

    def gaussian_pdf(self, x, mean, variance):
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean) ** 2 / (2 * variance))

    def predict(self, X):
        predictions = []

        for x in X:
            # loop one row at a time
            probabilities = []
            for i, c in enumerate(self.classes):
                prior = np.log(self.class_probabilities[i])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.mean[i], self.variance[i])))
                posterior = prior + likelihood
                probabilities.append(posterior)
            predictions.append(self.classes[np.argmax(probabilities)])
        
        return predictions
    
###########################################################

class CategoricalNB:
    '''
    Build a naive bayes classifier from scratch using Python (native python and numpy). 

    Categorical Naive Bayes.

    Assumptions:
    1. Features are categorical in nature or converted into categorical
        Categories of each feature are mapped to [0,1,2,...]
    2. Features are independent 
    '''
    def __init__(self):
        self.classes = None
        self.class_probabilities = None
        self.feature_probabilities = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probabilities = np.zeros(len(self.classes))
        self.feature_probabilities = {}

        for i, c in enumerate(self.classes):
            # Get rows corresponding to class c
            X_class = X[y==c]

            # Compute the class probability
            self.class_probabilities[i] = np.sum(y == c) / len(y)

            # Compute feature probabilities
            self.feature_probabilities[c] = []

            for feature in range(X.shape[1]):
                feature_counts = np.bincount(X_class[:, feature])
                total_count = np.sum(feature_counts)
                probabilities = feature_counts/total_count
                self.feature_probabilities[c].append(probabilities)
            
    def predict(self, X):
        predictions = []
        for x in X:
            # loop one row at a time
            probabilities = []  #posterior class probabilities 
            for i, c in enumerate(self.classes):
                prior = np.log(self.class_probabilities[i])
                likelihood = 0
                for feature, value in enumerate(x):
                    if value < len(self.feature_probabilities[c][feature]):  # handle out of range values 
                        likelihood += np.log(self.feature_probabilities[c][feature][value])
                    else:
                        likelihood =  float('-inf')
                
                posterior = prior + likelihood
                probabilities.append(posterior)
            predictions.append(self.classes[np.argmax(probabilities)])

        return predictions