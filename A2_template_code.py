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


# Import Data
data1 = pd.read_csv('prob1data.csv', header=None).values
X = data1[:, 0:2]
y = data1[:, -1]

# Hint: examine the data before you start coding SVM


def split_data(x, y):
    temp_x = []
    temp_y = []
    for i in range(len(y)):
        if y[i] == 1:
            temp_x.append(x[i].tolist())
        else:
            temp_y.append(x[i].tolist())
    return np.asarray(temp_x), np.asarray(temp_y)


# Problem (1a)
def LinearSVM_Primal(X, y, C, iter=10000):
    start_time = time.time()
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
    prob.solve(max_iter=iter)
    sol_time = time.time() - start_time
    print(prob.status)
    w = W.value
    b = b.value

    # Plot the data and the line.
    plt.scatter(cluster_1[:, 0], cluster_1[:, 1], color='blue')
    plt.scatter(cluster_2[:, 0], cluster_2[:, 1], color='green')
    x = np.linspace(-10, 10, 40)
    plt.plot(x, (-b - (w[0] * x)) / w[1], 'm')
    plt.show()

    return w, b, sol_time


# print(LinearSVM_Primal(X, y, 0))


# # Problem (1b)
def LinearSVM_Dual(X, y, C):
    cluster_1, cluster_2 = split_data(X, y)
    plt.scatter(cluster_1[:, 0], cluster_1[:, 1], color='blue')
    plt.scatter(cluster_2[:, 0], cluster_2[:, 1], color='green')
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    start_time = time.time()
    n, p = X.shape
    y = y.reshape(-1, 1) * 1.
    X_dash = y * X

    P = cvxopt_matrix(X_dash.dot(X_dash.T))
    q = cvxopt_matrix(-np.ones((n, 1)))
    G = cvxopt_matrix(np.vstack((-np.diag(np.ones(n)), np.identity(n))))
    h = cvxopt_matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)

    alphas = np.array(sol['x'])
    sol_time = time.time() - start_time
    w = ((y * alphas).T @ X).reshape(-1, 1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()

    # Computing b
    b = y[S] - np.dot(X[S], w)
    b = b[0]

    # Display results
    x = np.linspace(-1, 5, 20)
    plt.plot(x, (-b - (w[0] * x)) / w[1], 'm')
    plt.show()
    return alphas, sol_time


# print(LinearSVM_Dual(X, y, 1))


# Problem (1d)
def Linearly_separable (X, y):
    w, b, t = LinearSVM_Primal(X, y, 1000, iter=1000000)
    for i in range(len(X)):
        if y[i] * (np.matmul(w.T, X[i]) + b) <= 1:
            return 0
    return 1


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

np.random.seed(1)
n = 20
m = 1000
TEST = m
DENSITY = 0.2
beta_true = np.random.randn(n,1)
idxs = np.random.choice(range(n), int((1-DENSITY)*n), replace=False)
for idx in idxs:
    beta_true[idx] = 0
offset = 0
sigma = 45
X = np.random.normal(0, 5, size=(m,n))
Y = np.sign(X.dot(beta_true) + offset + np.random.normal(0,sigma,size=(m,1)))
X_test = np.random.normal(0, 5, size=(TEST,n))
Y_test = np.sign(X_test.dot(beta_true) + offset + np.random.normal(0,sigma,size=(TEST,1)))

cluster_1, cluster_2 = split_data(X, y)


for i in range(len(y)):
    if y[i] == -1:
        y[i] = 0
temp = np.squeeze(Y)
print(temp.shape)
print(Linearly_separable(X, temp))
