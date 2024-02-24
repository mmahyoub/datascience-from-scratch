'''
Implement decision tree algorithm from scratch. 

- DecisionTreeClassifier()
'''
import numpy as np 

class Node:
    '''Tree node class: A fundamental structure unit of the decision tree.'''
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        '''
        feature: index of the feature upon which splitting happens at the node 
        threshold: the splitting threshold, a value that determines how data is assigned to left and right nodes (< to left, otherwise to right)
        left: left branch from current node; it could be another internal node or a leaf node 
        right: right branch from current node; ut could be another internal node or a leaf node 
        value: represented for leaf nodes only; class label or the most occured label in the data assigned to leaf node
        '''
        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.value = value

class DecisionTreeClassifier():
    '''Classification based on the decision tree algorithm; decision tree constructor'''

    def __init__(self, max_depth = None):
        '''
        max_depth: maximum depth allowed; length of the longest path from the root node to a leaf node; key to conrol the complexity of the resulted tree
        root: A root node that will be built recursively when the fit method is called
        '''
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        '''
        X: A numpy array for features data 
        y: A numpy array for target data 
        '''
        assert isinstance(X, np.ndarray), 'X must be a numpy array'
        assert isinstance(y, np.ndarray), 'y must be a numpy array'

        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth = 0):
        '''
        An internal method for recursively constructing the decision tree.

        Arguments:
            X: A numpy array for features data. 
            y: A numpy array for target data. 
            depth: Current depth of the current node; when reaches max_depth, the tree construction will stop.
        Returns:
            Node: A Node object representing the root node of the subtree being constructed by current recursive call.
        '''

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Check if stopping criteria is met 
        if (self.max_depth is not None and depth >= self.max_depth) or (n_classes == 1):
            '''
            If stopping criterion is met a leaf node object is returned.
            . Max depth has reached, or 
            . One class data is presented 
            '''
            return Node(value=np.argmax(np.bincount(y)))

        # Find best split: feature index and threshold that best splits the data, maximizing information gain or minimizing impurity
        best_feature, best_threshold = self._find_best_split(X, y)

        # Perform splitting 
        left_indices = X[:, best_feature] < best_threshold
        X_left = X[left_indices]
        y_left = y[left_indices]
        X_right = X[~left_indices]
        y_right = y[~left_indices]

        # Build left and right subtrees recursivley 
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth +1)

        return Node(feature = best_feature, threshold = best_threshold, left = left_subtree, right = right_subtree)
    
    def _find_best_split(self, X, y):
        '''
        A helping function to find the best split feature index and threshold. 
        Minimizes gini impurity.
        Arguments:
            X: A numpy array for features data. 
            y: A numpy array for target data.
        Returns:
            best_feature: The index of the best feature for splitting. 
            best_threshold: Best splitting value.  
        '''

        n_samples, n_features = X.shape
        best_gini = float('inf')  # goal is to find splitting feature that minimizes this
        best_feature = None
        best_threshold = None 

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                gini = self._gini_impurity(y[left_indices], y[~left_indices])
                
                if gini < best_gini:
                    best_gini = gini 
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _gini_impurity(self, y_left, y_right):
        '''
        A helping function to compute the gini impurity.
        Gini impurity quantifies uncertainty or impurity in a dataset.
        A gini impurity of 0 means the set is pure; elements of set belong to the same class. 
        A gini impurity of 1 means maximum impurity; elements are evenly distributed across all classes.

        Arguments:
            y_left: Numpy array containing class labels of the left branch. 
            y_right: Numpy array containing class labels of the right branch.
        Returns:
            gini: gini measure of impurity 
        '''
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        gini_left = 1.0 - sum([(np.sum(y_left == c) / n_left)** 2 for c in np.unique(y_left)])
        gini_right = 1.0 - sum([(np.sum(y_right == c) / n_right)**2 for c in np.unique(y_right)])
        gini = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right
        return gini

    def predict(self, X):
        '''
        Generate predictions for a new instance. 
        Arguments:
            X: A numpy array of feature data. 
        Returns:
            predictions: predicted labels of instances in X.
        '''
        predictions = np.array([self._predict(x, self.root) for x in X])
        return predictions
    
    def _predict(self, x, node):
        '''
        A recursive helping function for traversing the decision tree to predict a label for x. 

        Arguments:
            x: Input sample; row of the X matrix 
            node: Current node being considered for prediction. 
        Returns:
            pred_label (node.value): The predicted label for x instance 
        '''
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._predict(x, node.left) 
        else:
            return self._predict(x, node.right)