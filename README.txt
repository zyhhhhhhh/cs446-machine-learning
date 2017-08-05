{\rtf1\ansi\ansicpg1252\cocoartf1404\cocoasubrtf470
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Yihao Zhang yzhng127@illinois.edu\
Codes are in data folder.\
Algorithm Description:\
1. SGD\
Use feature generation to generate arffs and run sgd.py\
\
2. DT(full,4,8)\
In test.sh uncomment the following lines for dt. And set tree depth in WeKaTester.java to be the corresponding depth:\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f1 \cf0 #java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train1.arff ./../badges.fold1.arff\uc0\u8232 #java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train2.arff ./../badges.fold2.arff\u8232 #java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train3.arff ./../badges.fold3.arff\u8232 #java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train4.arff ./../badges.fold4.arff\u8232 #java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train5.arff ./../badges.fold5.arff\
\
3.DT as features\
First, run generate_arff.py to get arffs in 5 folders called tree_1,tree_2\'85tree_5.\
Code for running DTs are in tree_100.java\
Second, uncomment the following comment in test.sh and run:\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_1/tree ./../badges.fold1.arff sgdout1.txt testout1.txt\uc0\u8232 java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_2/tree ./../badges.fold2.arff sgdout2.txt testout2.txt\u8232 java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_3/tree ./../badges.fold3.arff sgdout3.txt testout3.txt\u8232 java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_4/tree ./../badges.fold4.arff sgdout4.txt testout4.txt\u8232 java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_5/tree ./../badges.fold5.arff sgdout5.txt testout5.txt\
At last, run sgd_new.py and 5 accuracies will print in the console.\
I wrote a new sgd for the fifth algorithm because I parsed ARFF in the first one and for the second one I just created files not in arff for running sgd. Because I only need the features. \
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0
\cf0 \
}