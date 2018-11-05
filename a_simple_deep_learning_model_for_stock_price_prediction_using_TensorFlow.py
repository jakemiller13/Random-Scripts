#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:21:20 2018

@author: Jake
"""

'''
This is code from the article "A simple deep learning model for stock price
prediction using TensorFlow":

https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877

Data was already cleaned/prepared - LOCF'ed - and shifted 1 minute into future
'''

# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# Import and clean up/organize data
data = pd.read_csv('../Data/data_stocks_for_article.csv')
data = data.drop(['DATE'], 1)
n = data.shape[0] #rows
p = data.shape[1] #columns
plt.plot(data['SP500'])
plt.title('S&P 500')

# Convert to numpy array, create column variable to help visualize later
data_columns = data.columns
data = data.values

# Prepare train/test data
train_start = 0
train_end = int(np.floor(0.8 * n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data - NOTE: fit_transform and transform are achieving same result
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# Features and targets
X_train = data_train[:, 1:]
X_test = data_test[:, 1:]
y_train = data_train[:, 0]
y_test = data_test[:, 0]

# Model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Placeholders
X = tf.placeholder(dtype = tf.float32, shape = [None, n_stocks])
Y = tf.placeholder(dtype = tf.float32, shape = [None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(
                     mode = 'fan_avg',
                     distribution = 'uniform',
                     scale = sigma)
bias_initializer = tf.zeros_initializer()

# Variables
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layers
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))