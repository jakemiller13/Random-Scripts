# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:12:48 2018

@author: jmiller
"""

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
list(iris.keys())

X = iris['data'][:, 3:] # this is petal width
y = (iris['target'] == 2).astype(np.int) #1 if Iris-Virginica

log_reg = LogisticRegression(solver = 'lbfgs')
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

comb = np.column_stack((X[:, 0], y))
virginica = comb[comb[:, 1] == 1]
not_virginica = comb[comb[:, 1] == 0]

plt.plot(X_new, y_proba[:, 1], 'g-', label = 'Iris-Virginica')
plt.plot(X_new, y_proba[:, 0], 'b--', label = 'Not Iris-Virginica')
plt.plot(virginica[:, 0], virginica[:, 1], 'r^')
plt.plot(not_virginica[:, 0], not_virginica[:, 1], 'bs')
plt.legend(loc = 'best')
plt.xlabel('Petal width(cm)')
plt.xlim(0.0, 3.0)
plt.ylabel('Probability')