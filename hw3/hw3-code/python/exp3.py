import gen
import algorithms
import numpy as np
import matplotlib.pyplot as plt


def run_perceptron():
    best = [0,0,0,0,0,0]
    for n in range(3):
        w1, theta1, _ = algorithms.perceptron(d1y[n], d1x[n])
        correct1 = verify(w1,theta1,d2x[n],d2y[n])
        print("perceptron with n = "+str(m_val[n]))
        print(correct1)
        if correct1 > best[0]:
            best[0] = correct1
            best[1] = m_val[n]
    print("bestresult: correct1 = "+str(best[0])+" m = "+str(best[1]))


def run_perceptron_margin():
    r = [1.5,0.25,0.03,0.005,0.001]
    for n in range(3):
        best = [0, 0, 0, 0]
        for c in range(len(r)):
            w1, theta1, _ = algorithms.perceptron_m(d1y[n], d1x[n], r[c])
            correct1 = verify(w1, theta1, d2x[n], d2y[n])
            print("perceptron_margin with m = "+str(m_val[n])+", r = " + str(r[c]))
            print(correct1)
            if correct1 > best[0]:
                best[0] = correct1
                best[1] = r[c]
        print("bestresult for m = "+str(m_val[n])+" : correct1 = " + str(best[0]) +" learning rate = "+str(best[1]))


def run_winnow():
    alpha = [1.1,1.01,1.005,1.0005,1.0001]
    for n in range(3):
        best = [0, 0, 0, 0]
        for c in range(len(alpha)):
            w1, theta1, _ = algorithms.winnow(d1y[n], d1x[n], alpha[c],m_val[n])
            correct1 = verify(w1, theta1, d2x[n], d2y[n])
            print("winnow with m = "+str(m_val[n])+", alpha = " + str(alpha[c]))
            print(correct1)
            if correct1 > best[0]:
                best[0] = correct1
                best[1] = alpha[c]
        print("bestresult m = "+str(m_val[n])+": correct1 = " + str(best[0]) + " alpha = " + str(best[1]))


def run_winnow_margin():
    alpha = [1.1,1.01,1.005,1.0005,1.0001]
    gamma = [2.0,0.3,0.04,0.006,0.001]
    for n in range(3):
        best = [0, 0, 0, 0]
        for c in range(len(alpha)):
            for g in range(len(gamma)):
                w1, theta1, _ = algorithms.winnow_m(d1y[n], d1x[n], alpha[c], gamma[g],m_val[n])
                correct1 = verify(w1, theta1, d2x[n], d2y[n])
                print("winnow_margin with m = "+str(m_val[n])+", alpha = " + str(alpha[c])+" gamma = " + str(gamma[g]))
                print(correct1)
                if correct1 > best[0]:
                    best[0] = correct1
                    best[1] = alpha[c]
                    best[2] = gamma[g]
        print("bestresult for m = "+str(m_val[n])+": correct1 = " + str(best[0])+ " alpha = " + str(best[1])+" gamma = "+str(best[2]))


def run_adagrad():
    r = [1.5,0.25,0.03,0.005,0.001]
    for n in range(3):
        best = [0, 0, 0, 0]
        for c in range(len(r)):
            w1, theta1, _ = algorithms.adagrad(d1y[n], d1x[n], r[c])
            correct1 = verify(w1, theta1, d2x[n], d2y[n])
            print("adagrad with m = "+str(m_val[n])+", r = " + str(r[c]))
            print(correct1)
            if correct1 > best[0]:
                best[0] = correct1
                best[1] = r[c]
        print("bestresult for m = "+str(m_val[n])+": correct1 = " + str(best[0]) +" learning rate = " + str(best[1]))


def test_perceptron():
    for n in range(3):
        w1, theta1, _ = algorithms.perceptron(np.tile(d_y[n], 20), np.tile(d_x[n],(20,1)))
        correct1 = verify(w1,theta1,dxx[n],dyy[n])
        print("TEST perceptron with m = "+str(m_val[n]))
        print(correct1)


def test_perceptron_margin():
    r = [0.03,0.25,0.03]
    for n in range(3):
        w1, theta1, _ = algorithms.perceptron_m(np.tile(d_y[n], 20), np.tile(d_x[n],(20,1)), r[n])
        correct1 = verify(w1, theta1, dxx[n],dyy[n])
        print("TEST perceptron_margin with m = "+str(m_val[n])+", learning rate = " + str(r[n]))
        print(correct1)


def test_winnow():
    alpha = [1.01,1.1,1.1]
    for n in range(3):
        w1, theta1, _ = algorithms.winnow(np.tile(d_y[n], 20), np.tile(d_x[n],(20,1)), alpha[n],m_val[n])
        correct1 = verify(w1, theta1, dxx[n],dyy[n])
        print("TESTwinnow with m = "+str(m_val[n])+", alpha = " + str(alpha[n]))
        print(correct1)


def test_winnow_margin():
    alpha = [1.01,1.1,1.1]
    gamma = [2.0,0.006,0.001]
    for n in range(3):
        w1, theta1, _ = algorithms.winnow_m(np.tile(d_y[n], 20), np.tile(d_x[n],(20,1)), alpha[n], gamma[n],m_val[n])
        correct1 = verify(w1, theta1, dxx[n],dyy[n])
        print("TEST winnow_margin with m = "+str(m_val[n])+", alpha = " + str(alpha[n])+" gamma = " + str(gamma[n]))
        print(correct1)


def test_adagrad():
    r = [0.25,1.5,1.5]
    for n in range(3):
        w1, theta1, _ = algorithms.adagrad(np.tile(d_y[n], 20), np.tile(d_x[n],(20,1)), r[n])
        correct1 = verify(w1, theta1, dxx[n],dyy[n])
        print("TEST adagrad with m = "+str(0.25)+", r = " + str(r[n]))
        print(correct1)

# def test_bonus():

def verify(w,theta,dx_test,dy_test):
    correct = 0
    for i in range(len(dx_test)):
        if (np.inner(w,dx_test[i])+theta > 0 and dy_test[i] > 0) or (np.inner(w,dx_test[i])+theta <= 0 and dy_test[i] < 0):
            correct += 1
    return correct/len(dx_test)

m_val = [100,500,1000]
d_x = []
d_y =[]
d1x = []
d2x = []
d1y = []
d2y = []
dxx = []
dyy = []
for i in range(0,3):
    dy, dx = gen.gen(10, m_val[i],1000, 50000, True)
    d_y.append(dy)
    d_x.append(dx)
    d1x.append(np.tile(dx[0:5000],(20,1)))
    d2x.append(dx[5000:10000])
    d1y.append(np.tile(dy[0:5000], 20))
    d2y.append(dy[5000:10000])
    dyy_temp, dxx_temp = gen.gen(10, m_val[i], 1000, 10000, 0)
    dxx.append(dxx_temp)
    dyy.append(dyy_temp)
# run_perceptron()
# run_perceptron_margin()
# run_winnow()
# run_winnow_margin()
# run_adagrad()
# test_perceptron()
# test_perceptron_margin()
# test_winnow()
# test_winnow_margin()
test_adagrad()
