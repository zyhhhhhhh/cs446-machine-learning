import gen
import algorithms
import numpy as np
import matplotlib.pyplot as plt


def adagrad(y,x,r):
    w = np.zeros(len(x[0])+1)
    G = np.ones(len(x[0])+1)
    error = []
    errcount = 0
    Q = []
    q = 0
    for i in range(len(x)):
        if y[i]*(np.inner(w[0:-1],x[i])+w[-1]) <= 1:
            errcount += 1
            q += max(0, 1- y[i]*(np.inner(w[0:-1],x[i])+w[-1]))
            for j in range(len(x[0])):
                G[j] += (-y[i]*x[i][j])**2
            G[-1] += y[i]**2
            x_new = np.append(x[i],1)
            for j in range(len(w)):
                w[j] += r*y[i]*np.divide(x_new[j], np.power(G[j], 0.5))
        error.append(errcount)
        Q.append(q)
    return w[0:-1], w[-1], error, Q

def test_bonus():
    w1, theta1, error_t,Q = adagrad(np.tile(dy, 50), np.tile(dx, (50, 1)), 1.5 )
    index = np.linspace(0, 50 * 10000, num=50, endpoint=False, dtype=int)
    # print(len(index))
    error_plot = [error_t[i] for i in index]
    for i in range(49,0,-1):
        error_plot[i] = error_plot[i]-error_plot[i-1]
    plt.plot(np.linspace(1,50,dtype = int),error_plot,color="blue")
    plt.xlabel("round")
    plt.ylabel("# of mistakes")
    plt.title("AdaGrad mistakes over rounds")
    plt.show()
    plt.figure()
    Q_plot = [Q[i] for i in index]
    for i in range(49, 0, -1):
        Q_plot[i] = Q_plot[i] - Q_plot[i - 1]
    plt.plot(np.linspace(1, 50, dtype=int), Q_plot, color="blue")
    plt.xlabel("round")
    plt.ylabel("hinge loss")
    plt.title("hinge loss over rounds")
    plt.show()

dy, dx = gen.gen(10, 20,40, 10000, True)

test_bonus()

