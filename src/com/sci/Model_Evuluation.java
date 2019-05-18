package com.sci;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.evaluation.MultilabelMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

class Model_Evuluation implements java.io.Serializable {

    static double SVM_Precision(int iteration, String datapath, double label, JavaSparkContext javasparkcontext, HashingTF tf){


        JavaRDD<LabeledPoint> labeledraw = ML_Models.labelingdata(datapath,label,javasparkcontext,tf);
        JavaRDD<LabeledPoint>[] splits = labeledraw.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

        SVMModel model = SVMWithSGD.train(training.rdd(), iteration);
        model.setThreshold(0.5);

        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            @Override
            public Tuple2<Double, Double> call(LabeledPoint p) {
                return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
            }
        });
        double accuracy = 100.0* predictionAndLabel.filter(
                new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        //System.out.println(pl._1() + " -- " + pl._2());
                        return pl._1().intValue() == pl._2().intValue();
                    }
                }).count() / (double)test.count();

        return  accuracy;
    }

    static double SVM_Precision_New(int iteration,JavaRDD<LabeledPoint> datalabel){

        JavaRDD<LabeledPoint>[] splits = datalabel.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

        SVMModel model = SVMWithSGD.train(training.rdd(), iteration);
        model.setThreshold(0.5);

        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            @Override
            public Tuple2<Double, Double> call(LabeledPoint p) {
                return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
            }
        });
        double accuracy = 100.0* predictionAndLabel.filter(
                new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        //System.out.println(pl._1() + " -- " + pl._2());
                        return pl._1().intValue() == pl._2().intValue();
                    }
                }).count() / (double)test.count();

        return  accuracy;
    }

    static double NB_Accuaricy(String datapath, double label, JavaSparkContext javasparkcontext, HashingTF tf, double lamda){

        JavaRDD<LabeledPoint> labeleddata = ML_Models.labelingdata(datapath,label,javasparkcontext,tf);
        JavaRDD<LabeledPoint>[] tmp = labeleddata.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set


        NaiveBayesModel model =  ML_Models.NB_Model(training,lamda);
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        return 100*metrics.accuracy();

    }

    static double NB_Accuaricy_bypass (JavaRDD<LabeledPoint> datalabel, double lamda){

        JavaRDD<LabeledPoint>[] tmp = datalabel.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set


        NaiveBayesModel model =  ML_Models.NB_Model(training,lamda);
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        return 100*metrics.accuracy();

    }
}
