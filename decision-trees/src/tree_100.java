package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.io.PrintWriter;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;

public class tree_100 {

  public static void main(String[] args) throws Exception {

    if (args.length != 5) {
      System.err.println("Usage: WekaTester arff-file");
      System.exit(-1);
    }
    Instances data = new Instances(new FileReader(new File(args[0])));
    Instances test = new Instances(new FileReader(new File(args[2])));
    double [][] sgd_mat = new double[294][100];
    double [][] test_mat = new double[test.numInstances()][100];
    for(int i = 0;i<100;i++){
      // Load the data
      Instances train = new Instances(new FileReader(new File(args[1]+i)));
      // The last attribute is the class label
      data.setClassIndex(data.numAttributes() - 1);
      train.setClassIndex(train.numAttributes() - 1);
      test.setClassIndex(test.numAttributes() - 1);

      // Create a new ID3 classifier. This is the modified one where you can
      // set the depth of the tree.
      Id3 classifier = new Id3();

      // An example depth. If this value is -1, then the tree is grown to full
      // depth.
      classifier.setMaxDepth(4);

      // Train
      classifier.buildClassifier(train);
      Evaluation evaluation = new Evaluation(data);
      double[] predictions = evaluation.evaluateModel(classifier,data);
      Evaluation evaluation1 = new Evaluation(test);
      double [] predictions1 = evaluation1.evaluateModel(classifier,test);
      for(int j = 0;j<predictions.length;j++){
        sgd_mat[j][i] = predictions[j] == 1.0 ? 1:-1;
      }
      for(int j = 0;j<predictions1.length;j++){
        test_mat[j][i] = predictions1[j] == 1.0 ? 1:-1;
      }
      // Print the classfier
//      System.out.println(classifier);
//      System.out.println();
//      System.out.println(evaluation.toSummaryString());
    }
    PrintWriter writer = new PrintWriter(args[3]);
    for(int x =0;x<294;x++){
      for(int y = 0;y<100;y++) {
        writer.print(sgd_mat[x][y]);
        writer.print(";");
      }
    }
    writer.close();
    PrintWriter writer1 = new PrintWriter(args[4]);
    for(int x =0;x<test.numInstances();x++){
      for(int y = 0;y<100;y++){
        writer1.print(test_mat[x][y]);
        writer1.print(";");
      }
    }
    writer1.close();

  }
}

