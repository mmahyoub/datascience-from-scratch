'''
Test the decision tree module.
. Run DecisionTreeClassifier on the iris dataset.
'''
import os 
from decision_tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging 

def main():
    set_logging()  #initialize logging 
    logging.info('Testing the DecisionTreeClassifier algorithm..')

    try:
        iris = load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 42)

        dt_clf = DecisionTreeClassifier(max_depth = 3)
        dt_clf.fit(X_train, y_train)
        
        y_pred = dt_clf.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        
        logging.info(f'Testing succeeded! Accuracy on the iris test samples is {accuracy}%')
    except Exception:
        logging.error('Testing failed: ', exc_info = True)

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