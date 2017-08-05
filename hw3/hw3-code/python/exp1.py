import gen
import algorithms
import numpy as np
import matplotlib.pyplot as plt

def run_perceptron():
    best = [0,0,0,0]
    w_500, theta_500, _ = algorithms.perceptron(D1_500_y, D1_500_x)
    w_1000, theta_1000, _ = algorithms.perceptron(D1_1000_y, D1_1000_x)
    correct_500 = verify(w_500,theta_500,D2_500_x,D2_500_y)
    print("perceptron with n = 500")
    print(correct_500)
    correct_1000 = verify(w_1000, theta_1000, D2_1000_x, D2_1000_y)
    print("perceptron with n = 1000")
    print(correct_1000)
    if (correct_500+correct_1000)/2.0 > (best[0]+best[1])/2.0:
        best[0] = correct_500
        best[1] = correct_1000
    print("bestresult: correct_500 = "+str(best[0])+" correct_1000 = "+str(best[1]))


def run_perceptron_margin():
    r = [1.5,0.25,0.03,0.005,0.001]
    best = [0, 0, 0, 0]
    for c in range(len(r)):
        w_500, theta_500, _ = algorithms.perceptron_m(D1_500_y, D1_500_x, r[c])
        w_1000, theta_1000, _ = algorithms.perceptron_m(D1_1000_y, D1_1000_x,r[c])
        correct_500 = verify(w_500, theta_500, D2_500_x, D2_500_y)
        print("perceptron(margin) with n = 500, r = " + str(r[c]))
        print(correct_500)
        correct_1000 = verify(w_1000, theta_1000, D2_1000_x, D2_1000_y)
        print("perceptron(margin) with n = 1000, r = " + str(r[c]))
        print(correct_1000)
        if (correct_500 + correct_1000) / 2.0 > (best[0] + best[1]) / 2.0:
            best[0] = correct_500
            best[1] = correct_1000
            best[2] = r[c]
    print("bestresult: correct_500 = " + str(best[0]) + " correct_1000 = " + str(best[1])+" learning rate = "+str(best[2]))

def run_winnow():
    alpha = [1.1,1.01,1.005,1.0005,1.0001]
    best = [0, 0, 0, 0]
    for c in range(len(alpha)):
        w_500, theta_500, _ = algorithms.winnow(D1_500_y, D1_500_x, alpha[c],500)
        w_1000, theta_1000, _ = algorithms.winnow(D1_1000_y, D1_1000_x,alpha[c],1000)
        correct_500 = verify(w_500, theta_500, D2_500_x, D2_500_y)
        print("winnow with n = 500, alpha = " + str(alpha[c]))
        print(correct_500)
        correct_1000 = verify(w_1000, theta_1000, D2_1000_x, D2_1000_y)
        print("winnow with n = 1000, alpha = " + str(alpha[c]))
        print(correct_1000)
        if (correct_500 + correct_1000) / 2.0 > (best[0] + best[1]) / 2.0:
            best[0] = correct_500
            best[1] = correct_1000
            best[2] = alpha[c]
    print("bestresult: correct_500 = " + str(best[0]) + " correct_1000 = " + str(best[1]) + " alpha = " + str(best[2]))


def run_winnow_margin():
    alpha = [1.1,1.01,1.005,1.0005,1.0001]
    gamma = [2.0,0.3,0.04,0.006,0.001]
    best = [0, 0, 0, 0]
    for c in range(len(alpha)):
        for g in range(len(gamma)):
            w_500, theta_500, _ = algorithms.winnow_m(D1_500_y, D1_500_x, alpha[c], gamma[g],500)
            w_1000, theta_1000, _ = algorithms.winnow_m(D1_1000_y, D1_1000_x,alpha[c], gamma[g],1000)
            correct_500 = verify(w_500, theta_500, D2_500_x, D2_500_y)
            print("winnow(margin) with n = 500, alpha = " + str(alpha[c])+" gamma = " + str(gamma[g]))
            print(correct_500)
            correct_1000 = verify(w_1000, theta_1000, D2_1000_x, D2_1000_y)
            print("winnow(margin) with n = 1000, alpha = " + str(alpha[c])+" gamma = " + str(gamma[g]))
            print(correct_1000)
            if (correct_500 + correct_1000) / 2.0 > (best[0] + best[1]) / 2.0:
                best[0] = correct_500
                best[1] = correct_1000
                best[2] = alpha[c]
                best[3] = gamma[g]
    print("bestresult: correct_500 = " + str(best[0]) + " correct_1000 = " + str(best[1]) + " alpha = " + str(best[2])+" gamma = "+str(best[3]))

def run_adagrad():
    r = [1.5,0.25,0.03,0.005,0.001]
    best = [0, 0, 0, 0]
    for c in range(len(r)):
        w_500, theta_500, _ = algorithms.adagrad(D1_500_y, D1_500_x, r[c])
        w_1000, theta_1000, _ = algorithms.adagrad(D1_1000_y, D1_1000_x,r[c])
        correct_500 = verify(w_500, theta_500, D2_500_x, D2_500_y)
        print("adagrad with n = 500, alpha = " + str(r[c]))
        print(correct_500)
        correct_1000 = verify(w_1000, theta_1000, D2_1000_x, D2_1000_y)
        print("adagrad with n = 1000, alpha = " + str(r[c]))
        print(correct_1000)
        if (correct_500 + correct_1000) / 2.0 > (best[0] + best[1]) / 2.0:
            best[0] = correct_500
            best[1] = correct_1000
            best[2] = r[c]
    print("bestresult: correct_500 = " + str(best[0]) + " correct_1000 = " + str(best[1]) + " learning rate = " + str(best[2]))

def plot_mistake():
    _,__, error1 = algorithms.perceptron(D_500_y,D_500_x)
    _, __, error2 = algorithms.perceptron_m(D_500_y, D_500_x,0.005)
    _,__, error3 = algorithms.winnow(D_500_y,D_500_x,1.1,500)
    _, __, error4 = algorithms.winnow_m(D_500_y, D_500_x, 1.1,2.0,500)
    _,__, error5 = algorithms.adagrad(D_500_y,D_500_x,0.25)
    p1, = plt.plot(error1,color = "blue", label = "perceptron")
    p2, = plt.plot(error2,color = "red", label = "perceptron with margin")
    p3, = plt.plot(error3,color = "orange", label = "winnow")
    p4, = plt.plot(error4, color="green", label="winnow with margin")
    p5, = plt.plot(error5, color="black", label="adagrad")
    plt.legend(handles=[p1, p2, p3,p4,p5],loc = 2)
    plt.title("mistake bound n=500")
    plt.show()

def plot_mistake_1000():
    _,__, error1 = algorithms.perceptron(D_1000_y,D_1000_x)
    _, __, error2 = algorithms.perceptron_m(D_1000_y, D_1000_x,0.005)
    _,__, error3 = algorithms.winnow(D_1000_y,D_1000_x,1.1,1000)
    _, __, error4 = algorithms.winnow_m(D_1000_y, D_1000_x, 1.1,2.0,1000)
    _,__, error5 = algorithms.adagrad(D_1000_y,D_1000_x,0.25)
    p1, = plt.plot(error1,color = "blue", label = "perceptron")
    p2, = plt.plot(error2,color = "red", label = "perceptron with margin")
    p3, = plt.plot(error3,color = "orange", label = "winnow")
    p4, = plt.plot(error4, color="green", label="winnow with margin")
    p5, = plt.plot(error5, color="black", label="adagrad")
    plt.legend(handles=[p1, p2, p3,p4,p5],loc = 2)
    plt.title("mistake bound n=1000")
    plt.show()

def verify(w,theta,d2x,d2y):
    correct = 0
    for i in range(len(d2x)):
        if (np.inner(w,d2x[i])+theta > 0 and d2y[i] > 0) or (np.inner(w,d2x[i])+theta <= 0 and d2y[i] < 0):
            correct += 1
    return correct/len(d2x)
D_500_y, D_500_x = gen.gen(10, 100, 500, 50000, 0)
D_1000_y, D_1000_x = gen.gen(10,100,1000,50000,0)
D1_500_x = np.tile(D_500_x[0:10000],(20,1))
D2_500_x = D_500_x[10000:20000]
D1_500_y = np.tile(D_500_y[0:10000],20)
D2_500_y = D_500_y[10000:20000]
D1_1000_x = np.tile(D_1000_x[0:10000],(20,1))
D2_1000_x = D_1000_x[10000:20000]
D1_1000_y = np.tile(D_1000_y[0:10000],20)
D2_1000_y = D_1000_y[10000:20000]
# for n == 500,D1_500_x, D1_500_y, D2_500_x, D2_500_y
# for n = 1000, D1_1000_x, D1_1000_y, D2_1000_x, D2_1000_y
# run_perceptron()
# run_perceptron_margin()
# run_winnow()
# run_winnow_margin()
# run_adagrad()
# plot_mistake()
plot_mistake_1000()
