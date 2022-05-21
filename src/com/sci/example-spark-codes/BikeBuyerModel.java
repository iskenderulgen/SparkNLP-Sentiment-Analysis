package spark.sparkmlib;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

public class BikeBuyerModel {

	public static void main(String[] args) {

		System.setProperty("hadoop.home.dir",
				"C:\\Users\\Serkan\\Desktop\\hadoop-common-2.2.0-bin-master");

		JavaSparkContext context = new JavaSparkContext("local", "App");

		JavaRDD<String> bikeRdd = context
				.textFile("C:\\Users\\Serkan\\Desktop\\data\\bikebuyers\\*");

		JavaRDD<LabeledPoint> data = bikeRdd.map(new Function<String, LabeledPoint>() {
					@Override
					public LabeledPoint call(String line) throws Exception {
						Bike bike = new Bike(line.split(","));
						double val = "Yes".equals(bike.getBikeBuyer()) ? 1.0 : 0.0;
						LabeledPoint LP = new LabeledPoint(val, bike.features());
						return LP;
					}
		});
		
		JavaRDD<LabeledPoint>[] randomSplit = data.randomSplit(new double[]{0.8,0.2});
		
		JavaRDD<LabeledPoint> training = randomSplit[0].cache();
		JavaRDD<LabeledPoint> test = randomSplit[1].cache();
		
		final Integer numClasses = 2;
		final String impurity = "entropy";
		final Integer maxDepth = 20;
		final Integer maxBins = 34;
		
		final DecisionTreeModel model = DecisionTree.trainClassifier(training,
				numClasses, Bike.categoricalFeaturesInfo(), impurity,
				maxDepth, maxBins);
		
		
		test.take(10).forEach(x ->{
			double predict = model.predict(x.features());
			System.out.println("Buyed : (Yes or no)" + predict + "  Real : " + x.label());
			
		});
		
		
		

		
		
		
		

	}

}
