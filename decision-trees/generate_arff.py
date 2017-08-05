from scipy.io import arff
import numpy as np
import itertools
import random
f = open('../badges.empty.arff', 'r')
meta_list= f.read()
for k in range(1,6):
    data, meta = arff.loadarff("../badges.train"+str(k)+".arff")
    data_f = []
    for i in range(len(data)):
        temp = []
        for j in range(0, len(data[0]) - 1):
            if data[i][j] == b"0":
                temp.append("0")
            else:
                temp.append("1")
        if data[i][len(data[0]) - 1] == b"+":
            temp.append("+")
        else:
            temp.append("-")
        data_f.append(temp)
    for counter in range(100):
        len_sample = int(0.5*len(data_f))
        output = random.sample(data_f,len_sample)
        f2 = open("../tree_"+str(k)+"/tree"+str(counter), 'w')
        for item in meta_list:
            f2.write(item)
        for i in output:
            for j in i:
                if j == "+" or j =="-":
                    f2.write(j)
                else:
                    f2.write(j)
                    f2.write(",")
            if output.index(i) < len(output)-1:
                f2.write('\n')