
# Random Forest Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris


# X, y = Load Dataset Here @Faizan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# <<<<<----------------------------------->>>>>



# Random Forest Model without python package 


import numpy as np
import random
from collections import Counter



def train_test_split(X, y, test_size):
    indices = list(range(len(X)))
    test_indices = random.sample(indices, int(len(X) * test_size))
    train_indices = list(set(indices) - set(test_indices))
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def calculate_gini(y):
    counter = Counter(y)
    total_instances = len(y)
    gini = 1.0
    for label, count in counter.items():
        prob = count / total_instances
        gini -= prob ** 2
    return gini

def split_dataset(X, y, feature, threshold):
    left_X = []
    right_X = []
    left_y = []
    right_y = []
    
    for i in range(len(X)):
        if X[i][feature] <= threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
            
    return np.array(left_X), np.array(right_X), np.array(left_y), np.array(right_y)

def best_split(X, y):
    best_gini = float("inf")
    best_feature = None
    best_threshold = None
    for feature in range(X.shape[1]):
        thresholds = set(X[:, feature])
        for threshold in thresholds:
            left_X, right_X, left_y, right_y = split_dataset(X, y, feature, threshold)
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            gini = (len(left_y) / len(y)) * calculate_gini(left_y) + (len(right_y) / len(y)) * calculate_gini(right_y)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold




class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            self.tree = Counter(y).most_common(1)[0][0]
        else:
            feature, threshold = best_split(X, y)
            if feature is None:
                self.tree = Counter(y).most_common(1)[0][0]
            else:
                left_X, right_X, left_y, right_y = split_dataset(X, y, feature, threshold)
                self.tree = {
                    "feature": feature,
                    "threshold": threshold,
                    "left": DecisionTree(self.max_depth),
                    "right": DecisionTree(self.max_depth),
                }
                self.tree["left"].fit(left_X, left_y, depth + 1)
                self.tree["right"].fit(right_X, right_y, depth + 1)

    def predict(self, x):
        if isinstance(self.tree, dict):
            feature = self.tree["feature"]
            threshold = self.tree["threshold"]
            if x[feature] <= threshold:
                return self.tree["left"].predict(x)
            else:
                return self.tree["right"].predict(x)
        else:
            return self.tree



class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = [DecisionTree(max_depth) for _ in range(n_estimators)]

    def fit(self, X, y):
        for tree in self.trees:
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        predictions = np.array([tree.predict(x) for tree in self.trees for x in X])
        return [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(len(X))]


# X, y = Load Dataset Here @Faizan

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForest(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy}')
