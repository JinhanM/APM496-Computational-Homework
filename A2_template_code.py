# Optimization in Machine Learning (Winter 2020)
# Assignment 2
# Template Code


# Before you start, please read the instructions of this assignment
# For any questions, please email to yhe@mie.utoronto.ca
# For free-response parts, please submit a seperate .pdf file

# Your Name: Jinhan Mei
# Email: jinhan.mei@mail.utoronto.ca

"""
Problem 1: Linear Support Vector Machine
"""

# Import Libraries
import numpy as np
import pandas as pd
import cvxpy as cp
import time
import cvxopt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pylab as plt
from datetime import datetime


# Import Data
data1 = pd.read_csv('prob1data.csv', header=None).values
X = data1[:, 0:2]
y = data1[:, -1]

# Hint: examine the data before you start coding SVM


def split_data(x, y):
    temp_x = []
    temp_y = []
    for i in range(len(y)):
        if y[i] == 0:
            temp_x.append(x[i].tolist())
        else:
            temp_y.append(x[i].tolist())
    return np.asarray(temp_x), np.asarray(temp_y)


# Problem (1a)
def LinearSVM_Primal(X, y, C):
    start = datetime.now()
    cluster_1, cluster_2 = split_data(X, y)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    D = X.shape[1]
    N = X.shape[0]
    W = cp.Variable(D)
    b = cp.Variable()
    loss = (0.5 * cp.sum_squares(W) + C * cp.sum(cp.pos(1 - cp.multiply(y, X * W + b))))
    prob = cp.Problem(cp.Minimize(loss / N))
    prob.solve()
    print(prob.status)
    w = W.value
    b = b.value

    # Plot the data and the line.
    plt.scatter(cluster_1[:, 0], cluster_1[:, 1], color='blue')
    plt.scatter(cluster_2[:, 0], cluster_2[:, 1], color='green')
    x = np.linspace(-1, 5, 20)
    plt.plot(x, (-b - (w[0] * x)) / w[1], 'm')
    plt.show()

    return w, b, datetime.now()-start


print(LinearSVM_Primal(X, y, 1))



# # Problem (1b)
#
# def LinearSVM_Dual (X, y, C):
#
#   n, p = X.shape
#   y = y.reshape(-1,1) * 1.
#   X_dash = y * X
#   # Complete the following code:
#   # cvxopt_solvers.qp(P, q, G, h, A, b)
#   # objective:    (1/2) x^T P x + q^T x
#   # constraints:  Gx < h
#   #               Ax = b
#   # example could be found here:
#   # https://cvxopt.org/userguide/coneprog.html#quadratic-programming
#
#   P = cvxopt_matrix(# -- INSERT YOUR CODE HERE -- #)
#   q = cvxopt_matrix(# -- INSERT YOUR CODE HERE -- #)
#   G = cvxopt_matrix(# -- INSERT YOUR CODE HERE -- #)
#   h = cvxopt_matrix(# -- INSERT YOUR CODE HERE -- #)
#   A = cvxopt_matrix(# -- INSERT YOUR CODE HERE -- #)
#   b = cvxopt_matrix(# -- INSERT YOUR CODE HERE -- #)
#
#   cvxopt_solvers.options['show_progress'] = False
#   cvxopt_solvers.options['abstol'] = 1e-10
#   cvxopt_solvers.options['reltol'] = 1e-10
#   cvxopt_solvers.options['feastol'] = 1e-10
#
#   # -- INSERT YOUR CODE HERE -- #
#   sol = cvxopt_solvers.qp(P, q, G, h, A, b)
#   # -- INSERT YOUR CODE HERE -- #
#   alphas = np.array(sol['x'])
#
#   return alphas, sol_time
#
#
# # Compute the decision boundary
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Compute the optimal support vectors
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Problem (1d)
# def Linearly_separable (X, y):
#   # -------- INSERT YOUR CODE HERE -------- #
#   #
#   #
#   # Output: sep = 1 if data linearly seperable
#   #         sep = 0 if data not linearly seperable
#
#   return sep
#
# # Problem (1f)
#
# def l2_norm_LinearSVM_Primal (X, y, C):
#   # -------- INSERT YOUR CODE HERE -------- #
#   #
#   #
#
#   return w, b, sol_time
#
# # Compute the decision boundary
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Compute the optimal support vectors
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Problem (1g)
#
# def l2_norm_LinearSVM_Dual (X, y, C):
#   zero_tol = 1e-7
#
#   cvxopt_solvers.options['show_progress'] = False
#   cvxopt_solvers.options['abstol'] = 1e-10
#   cvxopt_solvers.options['reltol'] = 1e-10
#   cvxopt_solvers.options['feastol'] = 1e-10
#
#   # -------- INSERT YOUR CODE HERE -------- #
#   #
#   #
#
#   sol = cvxopt_solvers.qp(P, q, G, h, A, b)
#
#   # -------- INSERT YOUR CODE HERE -------- #
#   #
#   #
#
#   return alphas, sol_time
#
# # Compute the decision boundary
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Compute the optimal support vectors
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Problem (1h)
#
# # Plot the decision boundaries and datapoints
# # -------- INSERT YOUR CODE HERE -------- #
#
# """
# Problem 2: Kernal Support Vector Machine and Application
# """
#
#
# # Import libraries
# from numpy import *
# import pandas as pd
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn import preprocessing
#
# import matplotlib.pylab as plt
#
# data2 = pd.read_csv('prob2data.csv',header=None).values
# X = data2[:,0:2]
# y = data2[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2020)
#
# # Problem (2a)
#
# def gaussian_kernal(sigma):
#     def gaussian_kernel_sigma(x1, x2):
#         # -------- INSERT YOUR CODE HERE -------- #
#         return # -------- INSERT YOUR CODE HERE -------- #
#     return gaussian_kernel_sigma
#
# # Problem (2b)
#
# kernel_SVM = SVC(# -- INSERT YOUR CODE HERE -- #)
#
# # Compute # of optimal support vectors
# # -- INSERT YOUR CODE HERE -- #
#
# # Compute prediction error ratio in test set
# # -- INSERT YOUR CODE HERE -- #
#
# # Plot the decision boundary with all datapoints
# # -- INSERT YOUR CODE HERE -- #
#
# # Import data for (2c) - (2e)
# data3 = pd.read_csv('votes.csv')
# X = data3[['white','black','poverty','density','bachelor','highschool','age65plus','income','age18under','population2014']]
# X = X.values
# X = preprocessing.scale(X)
#
# # Problem (2c)
# y = # -- INSERT YOUR CODE HERE -- #
#
# # Train / test split for (2d) - (2e)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 2020)
#
# # Problem (2d)
#
# # You may use SVC from sklearn.svm
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Compute # of optimal support vectors
# # -------- INSERT YOUR CODE HERE -------- #
#
# # Compute prediction error ratio in test set
# # -------- INSERT YOUR CODE HERE -------- #
