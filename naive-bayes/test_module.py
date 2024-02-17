'''
Test the module using a simple example. 
. Run GaussianNB on a simple example
. Run CategoricalNB on a simple example 
'''
import os
from naive_bayes import GaussianNB, CategoricalNB
import numpy as np
import logging

def main():
    set_logging()  #initialize logging 
    logging.info('Testing Gaussian Naive Bayes Classifier...')
    try:
        X = np.array([[1,2], [2,3], [4,5], [6,7]])
        y = np.array([0,0,1,1])
        gnb = GaussianNB()
        gnb.fit(X, y )
        X_test= np.array([[0.5,1.5], [8,9]])
        predictions = gnb.predict(X_test)
        logging.info(f'GNB is working and the generated predictions for\n{X_test} are:\n{predictions}')
    except Exception:
        logging.error('GNB failed: ', exc_info = True)
    
    logging.info('Testing Categorical Naive Bayes Classifier...')
    try:
        X = np.array([[0,1],[0,0],[1,0],[2,2],[2,3],[3,2]])
        y = np.array([0,0,0,1,1,1])
        cnb = CategoricalNB()
        cnb.fit(X, y)
        X_test = np.array([[1,1], [3,3], [0,3]])
        predictions = cnb.predict(X_test)
        logging.info(f'CNB is working and the generated predictions for \n{X_test} are:\n {predictions}')
    except Exception:
        logging.error('CNB failed: ', exc_info = True)

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