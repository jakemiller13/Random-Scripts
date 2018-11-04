# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:19:54 2018

@author: jmiller
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
X.sort(axis = 0)
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Try different degrees
degrees = [1, 2, 300]

poly_features_1 = PolynomialFeatures(degree = 1, include_bias = False)
X_poly_1 = poly_features_1.fit_transform(X)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X_poly_1, y)
lin_reg_1_values = lin_reg_1.coef_ * X + lin_reg_1.intercept_

poly_features_2 = PolynomialFeatures(degree = 2, include_bias = False)
X_poly_2 = poly_features_2.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_2, y)
lin_reg_2_values = lin_reg_2.coef_[0,1] * X**2 +\
    lin_reg_2.coef_[0,0] * X + lin_reg_2.intercept_

# Degree is so high it seems to be wiping out everything but intercept
poly_features_300 = PolynomialFeatures(degree = 300, include_bias = False)
X_poly_300 = poly_features_300.fit_transform(X)
lin_reg_300 = LinearRegression()
lin_reg_300.fit(X_poly_300, y)
lin_reg_300_values = []
for i in range(len(X)):
    value = 0
    for j in range(len(lin_reg_300.coef_[0])):
        value += lin_reg_300.coef_[0,j] * X[i,0] ** (j + 1)
    print('before: ' + str(type(value)))
    print('value before: ' + str(value))
    value += lin_reg_300.intercept_[0]
    print('after: ' + str(type(value)))
    print('value after: ' + str(value))
    lin_reg_300_values.append(value)

print('Degree: 1 || ' + 'Intercept: ' + str(lin_reg_1.intercept_) +
      ' || Coefficients: ' + str(lin_reg_1.coef_))

print('Degree: 2 || ' + 'Intercept: ' + str(lin_reg_2.intercept_) +
      ' || Coefficients: ' + str(lin_reg_2.coef_))

print('Degree: 300 || ' + 'Intercept: ' + str(lin_reg_300.intercept_) +
      ' || Coefficients: ' + str(lin_reg_300.coef_))

plt.scatter(X, y, color = 'b', marker = '^')
plt.plot(X, lin_reg_1_values, label = 'Degree: 1')
plt.plot(X, lin_reg_2_values, label = 'Degree: 2')
plt.plot(X, lin_reg_300_values, label = 'Degree: 300')
plt.ylim(0, 11)
plt.legend(loc = 'best')
plt.show()