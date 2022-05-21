package MLlib;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

import java.io.IOException;
import java.util.Arrays;

public class SVM_Dogrulama {
    public static final String DATA_PATH = "C:\\Users\\ULGEN\\Documents\\IdeaProjects\\Spark\\";

    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf().set("spark.driver.memory","20g").setMaster("local").setAppName("SVMDeneme");
        JavaSparkContext javaSparkContext = new JavaSparkContext(conf);
        //SparkContext sc = new SparkContext(conf);
        javaSparkContext.setLogLevel("ERROR");
        //sc.setLogLevel("ERROR");

        JavaRDD<String> badword = javaSparkContext.textFile(DATA_PATH + "keywords2\\badword\\badword.csv");

        final HashingTF tf = new HashingTF(20);

        JavaRDD<LabeledPoint> badwordex = badword.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("0a0d"); // kelimelere ayir
                return new LabeledPoint(1.0, tf.transform(Arrays.asList(keywords)));
            }
        });
        JavaRDD<LabeledPoint>[] splits = badwordex.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];
        SVMModel model = SVMWithSGD.train(training.rdd(), 100);
        //model.clearThreshold();
        model.setThreshold(0.5);

        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));


        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabels.rdd());

        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());

        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());

        JavaRDD<?> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.collect());

        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

        JavaRDD<?> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect());

        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());



        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            @Override
            public Tuple2<Double, Double> call(LabeledPoint p) {
                return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
            }
        });
        double accuracy = 1.0* predictionAndLabel.filter(
                new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        System.out.println(pl._1() + " -- " + pl._2());
                        return pl._1().intValue() == pl._2().intValue();
                    }
                }).count() / (double)test.count();
        System.out.println(" SVM accuracy : " + accuracy);
       // System.out.println(accuracy);

        /*JavaRDD<String> badword = javaSparkContext.textFile(DATA_PATH + "data\\Yorum1M.csv");

        final HashingTF tf = new HashingTF(20);

        JavaRDD<LabeledPoint> badwordex = badword.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("0a0d"); // kelimelere ayir
                return new LabeledPoint(1.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> training = badwordex.sample(false, 0.6, 11L);
        training.cache();
        JavaRDD<LabeledPoint> test = badwordex.subtract(training);

        int numIterations = 100;
        SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);
        model.clearThreshold();

        JavaPairRDD<Double, Double> predictionAndLabels = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            @Override
            public Tuple2<Double, Double> call(LabeledPoint p) {
                return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
            }
        });
        double accuracy = 1.0* predictionAndLabels.filter(
                new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        System.out.println(pl._1() + " -- " + pl._2());
                        return pl._1().intValue() == pl._2().intValue();
                    }
                }).count() / (double)test.count();
        System.out.println(" SVM accuracy : " + accuracy);
        System.out.println(accuracy);*/






    }
}
   /* */