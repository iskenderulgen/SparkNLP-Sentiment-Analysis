package com.sci;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.util.Arrays;

class ML_Models implements java.io.Serializable{

    static  SVMModel SVM_Model(int iteration, String datapath, double label, JavaSparkContext javasparkcontext, HashingTF tf){

        JavaRDD<LabeledPoint> labeleddata = labelingdata(datapath,label,javasparkcontext,tf);
        final SVMModel model = SVMWithSGD.train(labeleddata.rdd(), iteration);
        model.setThreshold(0.5);
        return model;
    }

    static  SVMModel SVM_Model_2(int iteration, JavaRDD<LabeledPoint> labeledraw){

        final SVMModel model = SVMWithSGD.train(labeledraw.rdd(), iteration);
        model.setThreshold(0.5);
        return model;
    }


    static NaiveBayesModel NB_Model(JavaRDD<LabeledPoint> labeledraw, double labmda) {
        NaiveBayesModel model = NaiveBayes.train(labeledraw.rdd(),labmda);
        return model;
    }


    static JavaRDD<LabeledPoint> labelingdata(String datapath, double label, JavaSparkContext javasparkcontext, HashingTF tf){

        JavaRDD<String> data = javasparkcontext.textFile(datapath);
        //in here we create sub function to split by line and label it.
        JavaRDD<LabeledPoint> labeleddata = data.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line)  {
                String[] keywords = line.split("0a0d");
                return new LabeledPoint(label,tf.transform(Arrays.asList(keywords)));
            }
        });
        MLUtils.saveAsLibSVMFile( labeleddata.rdd() ,  "C:\\Users\\ULGEN\\Desktop\\kjalala\\");
        System.out.println(labeleddata.getClass().getName());
        return labeleddata;

    }
}