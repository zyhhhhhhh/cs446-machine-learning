import gen
import algorithms
import numpy as np
import matplotlib.pyplot as plt
def run_perceptron():
    best = [0,0,0,0,0,0]
    for n in range(5):
        w1, theta1, _ = algorithms.perceptron(d1y[n], d1x[n])
        correct1 = verify(w1,theta1,d2x[n],d2y[n])
        print("perceptron with n = "+str(n_val[n]))
        print(correct1)
        if correct1 > best[0]:
            best[0] = correct1
            best[1] = n_val[n]
    print("bestresult: correct1 = "+str(best[0])+" n = "+str(best[1]))


def run_perceptron_margin():
    r = [1.5,0.25,0.03,0.005,0.001]
    for n in range(5):
        best = [0, 0, 0, 0]
        for c in range(len(r)):
            w1, theta1, _ = algorithms.perceptron_m(d1y[n], d1x[n], r[c])
            correct1 = verify(w1, theta1, d2x[n], d2y[n])
            print("perceptron_margin with n = "+str(n_val[n])+", r = " + str(r[c]))
            print(correct1)
            if correct1 > best[0]:
                best[0] = correct1
                best[1] = r[c]
        print("bestresult for n = "+str(n_val[n])+" : correct1 = " + str(best[0]) +" learning rate = "+str(best[1]))

def run_winnow():
    alpha = [1.1,1.01,1.005,1.0005,1.0001]
    for n in range(5):
        best = [0, 0, 0, 0]
        for c in range(len(alpha)):
            w1, theta1, _ = algorithms.winnow(d1y[n], d1x[n], alpha[c],n_val[n])
            correct1 = verify(w1, theta1, d2x[n], d2y[n])
            print("winnow with n = "+str(n_val[n])+", alpha = " + str(alpha[c]))
            print(correct1)
            if correct1 > best[0]:
                best[0] = correct1
                best[1] = alpha[c]
        print("bestresult n = "+str(n_val[n])+": correct1 = " + str(best[0]) + " alpha = " + str(best[1]))


def run_winnow_margin():
    alpha = [1.1,1.01,1.005,1.0005,1.0001]
    gamma = [2.0,0.3,0.04,0.006,0.001]
    for n in range(5):
        best = [0, 0, 0, 0]
        for c in range(len(alpha)):
            for g in range(len(gamma)):
                w1, theta1, _ = algorithms.winnow_m(d1y[n], d1x[n], alpha[c], gamma[g],n_val[n])
                correct1 = verify(w1, theta1, d2x[n], d2y[n])
                print("winnow_margin with n = "+str(n_val[n])+", alpha = " + str(alpha[c])+" gamma = " + str(gamma[g]))
                print(correct1)
                if correct1 > best[0]:
                    best[0] = correct1
                    best[1] = alpha[c]
                    best[2] = gamma[g]
        print("bestresult for n = "+str(n_val[n])+": correct1 = " + str(best[0])+ " alpha = " + str(best[1])+" gamma = "+str(best[2]))

def run_adagrad():
    r = [1.5,0.25,0.03,0.005,0.001]
    for n in range(5):
        best = [0, 0, 0, 0]
        for c in range(len(r)):
            w1, theta1, _ = algorithms.adagrad(d1y[n], d1x[n], r[c])
            correct1 = verify(w1, theta1, d2x[n], d2y[n])
            print("adagrad with n = "+str(n_val[n])+", r = " + str(r[c]))
            print(correct1)
            if correct1 > best[0]:
                best[0] = correct1
                best[1] = r[c]
        print("bestresult for n = "+str(n_val[n])+": correct1 = " + str(best[0]) +" learning rate = " + str(best[1]))

def plot_mistake():
    n_val = [40, 80, 120, 160, 200]
    p_with_m = [1.5,0.25,0.25,0.25,0.03]
    err= np.zeros((5,5))
    for i in range(0, 5):
        dyy, dxx = gen.gen(10, 20, n_val[i], 50000, 0)
        dxx = np.tile(dxx, (10, 1))
        dyy = np.tile(dyy,10)
        _,__, error1 = algorithms.perceptron(dyy, dxx)
        err[0][i] = count_converge(error1)
        _, __, error2 = algorithms.perceptron_m(dyy, dxx,p_with_m[i])
        err[1][i] = count_converge(error2)
        _,__, error3 = algorithms.winnow(dyy, dxx,1.1,n_val[i])
        err[2][i] = count_converge(error3)
        _, __, error4 = algorithms.winnow_m(dyy, dxx, 1.1,2.0,n_val[i])
        err[3][i] = count_converge(error4)
        _,__, error5 = algorithms.adagrad(dyy, dxx,1.5)
        err[4][i] = count_converge(error5)
    print(err)
    p1, = plt.plot(n_val,err[0],color = "blue", label = "perceptron")
    p2, = plt.plot(n_val,err[1],color = "red", label = "perceptron with margin")
    p3, = plt.plot(n_val,err[2],color = "orange", label = "winnow")
    p4, = plt.plot(n_val,err[3], color="green", label="winnow with margin")
    p5, = plt.plot(n_val,err[4], color="black", label="adagrad")
    plt.legend(handles=[p1, p2, p3,p4,p5],loc = 2)
    plt.title("Errors to converge")
    plt.show()

def count_converge(error):
    for i in range(1000,len(error)):
        if error[i] == error[i-1000]:
            return i
    return -1

def verify(w,theta,d2x,d2y):
    correct = 0
    for i in range(len(d2x)):
        if (np.inner(w,d2x[i])+theta > 0 and d2y[i] > 0) or (np.inner(w,d2x[i])+theta <= 0 and d2y[i] < 0):
            correct += 1
    return correct/len(d2x)


n_val = [40,80,120,160,200]
d1x = []
d2x = []
d1y = []
d2y = []
for i in range(0,5):
    dy, dx = gen.gen(10, 20, n_val[i], 50000, 0)
    d1x.append(np.tile(dx[0:10000],(20,1)))
    d2x.append(dx[10000:20000])
    d1y.append(np.tile(dy[0:10000], 20))
    d2y.append(dy[10000:20000])


# run_perceptron()
# run_perceptron_margin()
# run_winnow()
# run_winnow_margin()
# run_adagrad()
plot_mistake()