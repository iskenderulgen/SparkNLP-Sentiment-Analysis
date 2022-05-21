package spark.sparkmlib;

import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class NaiveByesSample {
	
	public static final String DATA_PATH = "C:\\Users\\Serkan\\Desktop\\data\\news20.tar\\20_newsgroup\\";

	public static void main(String[] args) {
		
		JavaSparkContext javaSparkContext = new JavaSparkContext("local","NaiveByesApp");
		
		JavaRDD<String> atheismFile = javaSparkContext.textFile(DATA_PATH + "alt.atheism\\*");
		JavaRDD<String> motorcyclesFile = javaSparkContext.textFile(DATA_PATH + "rec.motorcycles\\*");
		JavaRDD<String> comGraphicsFile = javaSparkContext.textFile(DATA_PATH + "comp.graphics\\*");
		
		
		final HashingTF tf = new HashingTF(1000);
		
		JavaRDD<LabeledPoint> atheismExamples = atheismFile.map(new Function<String, LabeledPoint>() {
			public LabeledPoint call(String line) {
				String[] keywords = line.split(" "); // kelimelere ayir
				return new LabeledPoint(5 ,tf.transform(Arrays.asList(keywords)));
			}
		});
		
		JavaRDD<LabeledPoint> motorcyclesExample = motorcyclesFile.map(new Function<String, LabeledPoint>() {
			public LabeledPoint call(String line) {
				String[] keywords = line.split(" "); // kelimelere ayir
				return new LabeledPoint(10 ,tf.transform(Arrays.asList(keywords)));
			}
		});
		
		JavaRDD<LabeledPoint> comGraphicsExample = comGraphicsFile.map(new Function<String, LabeledPoint>() {
			public LabeledPoint call(String line) {
				String[] keywords = line.split(" "); // kelimelere ayir
				return new LabeledPoint(15 ,tf.transform(Arrays.asList(keywords)));
			}
		});

		JavaRDD<LabeledPoint> fullRdd = atheismExamples.union(motorcyclesExample);
		JavaRDD<LabeledPoint> fullRdd1 = fullRdd.union(comGraphicsExample);
		
		NaiveBayesModel model = NaiveBayes.train(fullRdd1.rdd());
		
		
		
		
		String atheismWord = " The scenario you outline is reasonably "
				+ "consistent, but all the evidence that I am familiar with not only does"
				+ "not support it, but indicates something far different. The Earth, by"
				+ "latest estimates, is about 4.6 billion years old, and has had life for"
				+ "about 3.5 billion of those years. Humans have only been around for (at"
				+ "most) about 200,000 years. But, the fossil evidence inidcates that life"
				+ "has been changing and evolving, and, in fact, disease-ridden, long before"
				+ "there were people. (Yes, there are fossils that show signs of disease..."
				+ "mostly bone disorders, of course, but there are some.) Heck, not just"
				+ "fossil evidence, but what we've been able to glean from genetic study shows"
				+ "that disease has been around for a long, long time. If human sin was what"
				+ "brought about disease (at least, indirectly, though necessarily) then"
				+ "how could it exist before humans?";
		
		String compGraphicsWord =
				"I am looking to add voice input capability to a user interface I am " +
				"developing on an HP730 (UNIX) workstation. I would greatly appreciate " +
				"information anyone would care to offer about voice input systems that are " +
				"easily accessible from the UNIX environment. ";
		
		String compGraphicsWord2 = "Does anyone know of any good shareware animation or paint software for an SGI " +
		" machine?  I've exhausted everyplace on the net I can find and still don't hava " +
		" a nice piece of software. ";
		
		String compGraphicsWord3 = " pictures and films created using computers. Usually, the term refers to computer-generated " +
		"image data created with help from specialized graphical hardware and software. " +
		"It is a vast and recent area in computer science.";
		
		
		
		String motorcyclesWord =
				"When I got my knee rebuilt I got back on the street bike ASAP. I put " +
				"the crutches on the rack and the passenger seat and they hung out back a " +
				"LONG way. Just make sure they're tied down tight in front and no problemo. " ;
				
		String motorcyclesWord2 = "Whether this is your first motorcycle or you’ve been riding for years, you may want to start your  " +
		"search by deciding on a make and model. Yamaha and Honda Motorcycles are rugged and dependable.  " +
		"Or what about a zippy motorbike like a Kawasaki, Victory or Ducati? Don’t let your head spin—take  " +
				"our giant selection of motorcycles for sale one page at a time.  " +
				"Looking for an off-road motorcycle? How about one of the many great KTM models? ";
		
				
		
		
		Vector testAtheismWord = tf.transform(Arrays.asList(atheismWord.split(" ")));
		
		System.out.println("Prediction for atheismWord : " + model.predict(testAtheismWord));

		Vector testcompGraphicsWord = tf.transform(Arrays.asList(compGraphicsWord.split(" ")));
		
		System.out.println("Prediction for compGraphics : " + model.predict(testcompGraphicsWord));
		
		Vector testcompGraphicsWord2 = tf.transform(Arrays.asList(compGraphicsWord2.split(" ")));
		
		System.out.println("Prediction for compGraphics 2 : " + model.predict(testcompGraphicsWord2));
		
		Vector testcompGraphicsWord3 = tf.transform(Arrays.asList(compGraphicsWord3.split(" ")));
		
		System.out.println("Prediction for compGraphics 2 : " + model.predict(testcompGraphicsWord3));

		Vector testmotorcyclesWord = tf.transform(Arrays.asList(motorcyclesWord.split(" ")));
		
		System.out.println("Prediction for motorcyclesWord : " + model.predict(testmotorcyclesWord));

		Vector testmotorcyclesWord2 = tf.transform(Arrays.asList(motorcyclesWord2.split(" ")));
		
		System.out.println("Prediction for motorcyclesWord : " + model.predict(testmotorcyclesWord2));



		
		
		

	}

}
