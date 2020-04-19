import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# import data
data1 = pd.read_csv('prob1data.csv', header=None).values
X = data1[:, 0:2]
y = data1[:, -1]


def split_data(x, y):
    temp_x = []
    temp_y = []
    for i in range(len(y)):
        if y[i] == 0:
            temp_x.append(x[i].tolist())
        else:
            temp_y.append(x[i].tolist())
    return np.asarray(temp_x), np.asarray(temp_y)


def LinearSVM_Primal(X, y, C):
    # split and cluster the data frist.
    cluster_1, cluster_2 = split_data(X, y)
    plt.scatter(cluster_1[:, 0], cluster_1[:, 1], color='blue')
    plt.scatter(cluster_2[:, 0], cluster_2[:, 1], color='green')

    w = cp.Variable((2, 1))
    b = cp.Variable()

    error = cp.Variable()


    x_constraint = [w.T * cluster_1[i] + b >= 1 for i in range(len(cluster_1))]
    y_constraint = [w.T * cluster_2[i] + b <= -1 for i in range(len(cluster_2))]

    constraint = x_constraint + y_constraint

    error = sum([(w.T * cluster_1[i] + b)/(np.sqrt(cluster_1[i][0]**2 + cluster_1[i][1]**2)) for i in range(len(cluster_1))]) + sum([(w.T * cluster_2[i] + b)/(np.sqrt(cluster_2[i][0]**2 + cluster_2[i][1]**2)) for i in range(len(cluster_2))])

    obj = cp.Minimize(cp.norm(w, 2) + C * error)
    prob = cp.Problem(obj, constraint)

    prob.solve()
    print("Problem Status: %s" %prob.status)

    # Now, generate the line
    p = w.value
    q = b.value
    print(p)

    x = np.linspace(-1, 5, 20)
    plt.plot(x, (-q - (p[0]*x))/p[1])
    plt.show()

LinearSVM_Primal(X,y,1)
