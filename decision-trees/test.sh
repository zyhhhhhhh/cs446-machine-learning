#!/bin/bash

mkdir bin

make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the feature files may be built. Note that don't necessarily have to
# use Java for this step.

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.all ./../badges.example.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1 ./../badges.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2 ./../badges.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold3 ./../badges.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold4 ./../badges.fold4.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold5 ./../badges.fold5.arff
# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train1.arff ./../badges.fold1.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train2.arff ./../badges.fold2.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train3.arff ./../badges.fold3.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train4.arff ./../badges.fold4.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.train5.arff ./../badges.fold5.arff

java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_1/tree ./../badges.fold1.arff sgdout1.txt testout1.txt
java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_2/tree ./../badges.fold2.arff sgdout2.txt testout2.txt
java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_3/tree ./../badges.fold3.arff sgdout3.txt testout3.txt
java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_4/tree ./../badges.fold4.arff sgdout4.txt testout4.txt
java -cp lib/weka.jar:bin cs446.homework2.tree_100 ./../badges.example.arff ./../tree_5/tree ./../badges.fold5.arff sgdout5.txt testout5.txt
