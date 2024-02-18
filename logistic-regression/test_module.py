'''
Test the module using a simple example. 
. Run LogisticRegression on a simple example
'''
import os
from logistic_regression import LogisticRegression
import numpy as np
import logging

def main():
    set_logging()  #initialize logging 
    logging.info('Testing LogisticRegression algorithm...')
    
    try:
        X = np.array([[1,1], [1.5, 2], [2.5, 3], [4,1]])
        y = np.array([0,0,1,1])

        lr = LogisticRegression(learning_rate = 0.01, n_iterations = 1000)
        lr.fit(X,y)
        logging.info(f'Fitting completed. Learned parameters are: {[lr.intercept] + lr.coefficients.tolist()}')
        
        logging.info('Generating predictions...')
        X_test = np.array([[0.5,1], [5,2]])
        predictions = lr.predict(X_test)
        probabilities = lr.predict_proba(X_test)

        true_labels = [0,1]
        logging.info(f'Logistic Regression is working and the generated predictions for {X_test} with actual labels = {true_labels} are: {predictions} || probabilities = {probabilities.round(2)}')
    
    except Exception:
        logging.error('Logistic regression failed: ', exc_info = True)

def set_logging():
    if 'logs.log' in os.listdir(os.getcwd()):
        os.remove('logs.log')

    logging.basicConfig(level=logging.INFO)
    file_handler =  logging.FileHandler(filename = './logs.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

if __name__ == "__main__":
    main()