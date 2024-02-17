'''
Test the module using a simple example. 
. Run LinearRegression on a simple example
'''
import os
from linear_regression import LinearRegression
import numpy as np
import logging

def main():
    set_logging()  #initialize logging 
    logging.info('Testing LinearRegression algorithm...')
    try:
        X = np.array([[1,1], [2.5, 3], [4,1]])
        
        # Real parameters
        b = 1
        c1 = 2
        c2 = 3
        y = np.array([b + c1 * x + c2 * y for x,y in zip(X[:,0], X[:,1])])

        lr = LinearRegression(learning_rate = 0.01, n_iterations = 1000)
        lr.fit(X,y)
        logging.info(f'Fitting completed. Real parameters are: {[b,c1,c2]} and learned parameters are: {[lr.intercept] + lr.coefficients.tolist()}')
        
        logging.info('Generating predictions...')
        X_test = np.array([0.5,1])
        prediction = lr.predict(X_test)
        logging.info(f'Linear Regression is working and the generated prediction for {X_test} with actual y = {b + c1 * X_test[0] + c2 * X_test[1]} is: {prediction}')
    
    except Exception:
        logging.error('Linear regression failed: ', exc_info = True)

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