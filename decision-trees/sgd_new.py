from scipy.io import arff
import numpy as np
import itertools
def updatew(data,t,w,r):
    for i in range(len(data)):
        xd = data[i]
        xd = np.array(xd)
        od = np.inner(w,xd)
        deltaw = r*np.multiply((t[i]-od),xd)
        w = w+deltaw
    return w
for z in range(1,6):
    test = open("testout"+str(z)+".txt","r").read().split(";")[0:-1]
    sgd = open("sgdout"+str(z)+".txt","r").read().split(";")[0:-1]
    test_n = list(zip(*[iter(test)]*100))
    sgd_n = list(zip(*[iter(sgd)]*100))
    test_n = np.array(test_n,dtype = float)
    sgd_n = np.array(sgd_n,dtype=float)
    data_test, meta = arff.loadarff("../badges.fold"+str(z)+".arff")
    test_gold = []
    for i in range(len(data_test)):
        if data_test[i][len(data_test[0]) - 1] == b"+":
            test_gold.append(1)
        else:
            test_gold.append(-1)
    data_t, meta = arff.loadarff("../badges.example.arff")
    t = []
    for i in range(len(data_t)):
        if data_t[i][len(data_t[0]) - 1] == b"+":
            t.append(1)
        else:
            t.append(-1)
    r = 0.001
    w = np.zeros(len(sgd_n[0]))
    for i in range(1000):
        w = updatew(sgd_n,t, w, r)
    # test result
    correct = 0
    for i in range(len(test_n)):
        if (np.inner(test_n[i], w) >= 0 and test_gold[i] >= 0) or (np.inner(test_n[i], w) <= 0 and test_gold[i] <= 0):
            correct += 1
    print(correct / len(test_n))
