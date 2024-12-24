from __future__ import print_function

import os

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Data Directory')
args = parser.parse_args()
DATA_PATH = os.path.abspath(args.data_dir)
ARRAY_PATH = DATA_PATH+'/arrays/'

x_train, y_train = np.load(ARRAY_PATH+'x_train.npz')['arr_0'], np.load(ARRAY_PATH+'y_train.npz')['arr_0']
print("Loaded training data")
x_validation, y_validation = np.load(ARRAY_PATH+'x_validation.npz')['arr_0'], np.load(ARRAY_PATH+'y_validation.npz')['arr_0']
print("Loaded validation data")
x_test, y_test = np.load(ARRAY_PATH+'x_test.npz')['arr_0'], np.load(ARRAY_PATH+'y_test.npz')['arr_0']
print("Loaded testing data")

x_test = np.concatenate((x_validation, x_test), axis=0)
y_test = np.concatenate((y_validation, y_test), axis=0)

x_train = x_train.reshape(x_train.shape[0], -1)
y_train = np.argmax(y_train, axis=1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = np.argmax(y_test, axis=1)

names = np.array(['fish1', 'fish2', 'fish3', 'fish4', 'fish5'])

n_samples = y_train.shape[0]+y_test.shape[0]
n_features = x_train.shape[1]
n_classes = names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

n_components = 12
print("Extracting the top %d eigenvectors from %d sets" % (n_components, x_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(x_train)
print("Done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenvector's orthonormal basis")
t0 = time()
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print("done in %0.3fs" % (time() - t0))

def run_classifier(classifier, param_grid, verbose=0):
    t0 = time()
    clf = GridSearchCV(classifier, param_grid, n_jobs=-1, verbose=verbose)
    clf = clf.fit(x_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    t0 = time()
    y_pred = clf.predict(x_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

print("Random Forest")
forest_grid = {'n_estimators': [1, 3, 5, 10, 20, 50, 100, 250, 500],
               'max_depth': [1, 3, 5, 10, 20, 50, None]}
run_classifier(RandomForestClassifier(), forest_grid)

print("Gaussian Naive Bayes")
gnb_grid = {}
run_classifier(GaussianNB(), gnb_grid)

print("Gaussian Process")
gpc_grid = {'max_iter_predict': [1, 5, 10, 20, 50, 100, 200, 500, 1000]}
run_classifier(GaussianProcessClassifier(), gpc_grid)

print("K Nearest Neighbors")
knn_grid = {'n_neighbors': [1, 3 , 5, 10, 20],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'weights': ['uniform', 'distance'],
            'leaf_size': [1, 2, 5, 8, 10, 20, 30, 50, 100]}
run_classifier(KNeighborsClassifier(), knn_grid)

print("Quadratic Discriminant Analysis")
qda_grid = {}
run_classifier(QuadraticDiscriminantAnalysis(), qda_grid)

print("Support Vector Matrix")
svm_grid = {'C': [1, 5, 10, 20, 50, 1e3, 5e3, 1e4, 5e4, 1e5],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1], 
            'kernel': ['poly', 'rbf', 'sigmoid']}
run_classifier(SVC(), svm_grid)

print("Decision Tree")
tree_grid = {'max_depth': [1, 3, 5, 10, 20, 50, None]}
run_classifier(DecisionTreeClassifier(), tree_grid)

print("Multi-layer Perceptron")
mlp_grid = {'hidden_layer_sizes': [(100), (50,50), (100,100), (50,50,50), (50,100,50), (100,100,100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.00001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1],
        'learning_rate': ['constant','adaptive'],
        'max_iter': [1000],
        'early_stopping': [True]}
run_classifier(MLPClassifier(), mlp_grid, verbose=10)
