from scipy.io import arff
import numpy as np
import itertools
def updatew(data,w,r):
    for i in range(len(data)):
        xd = []
        for j in range(0, len(data[0])-1):
            if data[i][j] == b"0":
                xd.append(0)
            else:
                xd.append(1)
        xd = np.array(xd)
        od = np.inner(w,xd)
        if data[i][len(data[0])-1] == b"+":
            t = 1
        else:
            t = -1
        deltaw = r*np.multiply((t-od),xd)
        w = w+deltaw
    return w

def generate_test(data_test):
    test_num = []
    test_gold = []
    for i in range(len(data_test)):
        temp = []
        for j in range(0, len(data_test[0])-1):
            if data_test[i][j] == b"0":
                temp.append(0)
            else:
                temp.append(1)
        test_num.append(temp)
        if data_test[i][len(data_test[0])-1] == b"+":
            test_gold.append(1)
        else:
            test_gold.append(-1)
    return test_num, test_gold

def cross_validation(w,r,data_train,test_num,test_gold):
    for i in range(1000):
        w = updatew(data_train,w,r)
    correct = 0
    for i in range(len(test_num)):
        if (np.inner(test_num[i],w)>=0 and test_gold[i] >=0) or (np.inner(test_num[i],w)<=0 and test_gold[i] <=0):
            correct += 1
    retval = correct/len(test_num)
    return retval


data, meta = arff.loadarff("../badges.example.arff")
# create 5 fold
num_instance = len(data)
data1 = data[0:int(num_instance/5)]
data2 = data[int(num_instance/5):int(2*num_instance/5)]
data3 = data[int(2*num_instance/5):int(3*num_instance/5)]
data4 = data[int(3*num_instance/5):int(4*num_instance/5)]
data5 = data[int(4*num_instance/5):int(5*num_instance/5)]
train1 = list(itertools.chain(data2,data3,data4,data5))
train2 = list(itertools.chain(data1,data3,data4,data5))
train3 = list(itertools.chain(data1,data2,data4,data5))
train4 = list(itertools.chain(data1,data2,data3,data5))
train5 = list(itertools.chain(data1,data2,data3,data4))
test_num1,test_gold1 = generate_test(data1)
test_num2,test_gold2 = generate_test(data2)
test_num3,test_gold3 = generate_test(data3)
test_num4,test_gold4 = generate_test(data4)
test_num5,test_gold5 = generate_test(data5)
# initialize r,w,and do 5 fold
r = 0.001
w = np.zeros(len(data[0])-1)
result = []
result.append(cross_validation(w,r,train1,test_num1,test_gold1))
result.append(cross_validation(w,r,train2,test_num2,test_gold2))
result.append(cross_validation(w,r,train3,test_num3,test_gold3))
result.append(cross_validation(w,r,train4,test_num4,test_gold4))
result.append(cross_validation(w,r,train5,test_num5,test_gold5))
print(np.mean(result))