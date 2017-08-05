import numpy as np
def perceptron(y,x):
    w = np.zeros(len(x[0]))
    theta = 0
    r = 1
    error = []
    errcount = 0
    for i in range(len(x)):
        if y[i]*(np.inner(w,x[i])+theta) <= 0:
            errcount+=1
            w += r*y[i]*x[i]
            theta += r*y[i]
        error.append(errcount)
    return w,theta,error

def perceptron_m(y,x,r):
    w = np.zeros(len(x[0]))
    theta = 0
    error = []
    errcount = 0
    gamma = 1
    for i in range(len(x)):
        if y[i]*(np.inner(w,x[i])+theta) < gamma:
            errcount += 1
            w += r*y[i]*x[i]
            theta += r*y[i]
        error.append(errcount)
    return w,theta,error

def winnow(y,x,alpha,n):
    w = np.ones(len(x[0]))
    theta = -n
    error = []
    errcount = 0
    for i in range(len(x)):
        if y[i]*(np.inner(w,x[i])+theta) <= 0:
            errcount += 1
            for j in range(len(w)):
                w[j]*=alpha**(y[i]*x[i][j])
        error.append(errcount)
    return w,theta,error

def winnow_m(y,x,alpha,gamma,n):
    w = np.ones(len(x[0]))
    theta = -n
    error = []
    errcount = 0
    for i in range(len(x)):
        if y[i]*(np.inner(w,x[i])+theta) < gamma:
            errcount += 1
            for j in range(len(w)):
                w[j]*=alpha**(y[i]*x[i][j])
        error.append(errcount)
    return w,theta,error

def adagrad(y,x,r):
    w = np.zeros(len(x[0])+1)
    G = np.ones(len(x[0])+1)
    error = []
    errcount = 0
    for i in range(len(x)):
        if y[i]*(np.inner(w[0:-1],x[i])+w[-1]) <= 1:
            errcount += 1
            for j in range(len(x[0])):
                G[j] += (-y[i]*x[i][j])**2
            G[-1] += y[i]**2
            x_new = np.append(x[i],1)
            for j in range(len(w)):
                w[j] += r*y[i]*np.divide(x_new[j], np.power(G[j], 0.5))
        error.append(errcount)
    return w[0:-1], w[-1], error