#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:19:04 2018

@author: Jake
"""

'''
Chapter 5: Support Vector Machines
Hands-On Machine Learning with Scikit-Learn & TensorFlow
'''

#############################
# Linear SVM Classification #
#############################

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris['data'][:, (2,3)]
y = (iris['target'] == 2).astype(np.float64)

svm_clf = Pipeline([
                  ('scaler', StandardScaler()),
                  ('linear_svc', LinearSVC(C = 1, loss = 'hinge'))
                  ])

svm_clf.fit(X, y)
print('-- Linear SVM Classification --')
print('Predicting Iris-Virginica [0: Negative, 1: Positive]:')
dimensions = [5.5, 1.7]
print('Dimensions: ' + str(dimensions))
print(svm_clf.predict([dimensions]))

################################
# Nonlinear SVM Classification #
################################

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

moons = make_moons()
X = moons[0]
y = moons[1]

polynomial_svm_clf = Pipeline([
                             ('poly_features', PolynomialFeatures(degree = 3)),
                             ('scaler', StandardScaler()),
                             ('svm_clf', LinearSVC(C = 10, loss = 'hinge'))
                             ])
polynomial_svm_clf.fit(X, y)

#####################
# Polynomial Kernel #
#####################

poly_kernel_svm_clf = Pipeline([
                              ('scaler', StandardScaler()),
                              ('svm_clf', SVC(kernel = 'poly', degree = 3,
                                              coef0 = 1, C = 5))
                              ])
poly_kernel_svm_clf.fit(X, y)




