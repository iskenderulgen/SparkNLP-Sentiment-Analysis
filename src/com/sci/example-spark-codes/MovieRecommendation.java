package spark.sparkmlib;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.SQLContext;

import com.jasongoodwin.monads.Try;

public class MovieRecommendation {
	
	
	public static final String moviesPath = "C:\\Users\\Serkan\\Desktop\\data\\ml-1m\\movies.dat";
	public static final String usersPath = "C:\\Users\\Serkan\\Desktop\\data\\ml-1m\\users.dat";
	public static final String ratingsPath = "C:\\Users\\Serkan\\Desktop\\data\\ml-1m\\ratings.dat";

	public static void main(String[] args) {

		System.setProperty("hadoop.home.dir",
				"C:\\Users\\Serkan\\Desktop\\hadoop-common-2.2.0-bin-master");

		
		
		JavaSparkContext jsc = new JavaSparkContext("local", "Recommendation Engine");
		SQLContext sqlContext = new SQLContext(jsc);
		
		/*
		* Load Movie data
		*/
		JavaRDD<Movie> movieRDD = jsc.textFile(moviesPath).map(new Function<String, Movie>() {
			public Movie call(String line) throws Exception {
				String[] movieArr = line.split("::");
				Integer movieId = Integer.parseInt(Try.ofFailable(() -> movieArr[0]).orElse("-1"));
				return new Movie(movieId, movieArr[1], movieArr[2]);
			}
		}).cache();
		
		JavaRDD<User> userRDD = jsc.textFile(usersPath).map(new Function<String, User>() {
			@Override
			public User call(String line) throws Exception {
				String[] userArr = line.split("::");
				Integer userId = Integer.parseInt(Try.ofFailable(() -> userArr[0]).orElse("-1"));
				Integer age = Integer.parseInt(Try.ofFailable(() -> userArr[2]).orElse("-1"));
				Integer occupation = Integer.parseInt(Try.ofFailable(() -> userArr[3]).orElse("-1"));
				return new User(userId, userArr[1], age, occupation, userArr[4]);
			}
		}).cache();		
		
		JavaRDD<Rating> ratingRDD = jsc.textFile(ratingsPath).map(new Function<String, Rating>() {
			@Override
			public Rating call(String line) throws Exception {
				String[] ratingArr = line.split("::");
				Integer userId = Integer.parseInt(Try.ofFailable(() -> ratingArr[0]).orElse("-1"));
				Integer movieId = Integer.parseInt(Try.ofFailable(() -> ratingArr[1]).orElse("-1"));
				Double rating = Double.parseDouble(Try.ofFailable(() -> ratingArr[2]).orElse("-1"));
			return new Rating(userId, movieId, rating);
			}
		}).cache();
		
		System.out.println("Total number of movie : " + movieRDD.count());
		System.out.println("Total number of user : " + userRDD.count());
		System.out.println("Total number of rating : " + ratingRDD.count());
		
		JavaRDD<Rating>[] randomSplit = ratingRDD.randomSplit(new double[]{0.8,0.2});
		
		JavaRDD<Rating> trainingRDD = randomSplit[0].cache();
		JavaRDD<Rating> testRDD = randomSplit[1].cache();
		
		
		long countTraining = trainingRDD.count();
		long countTest = testRDD.count();
		
		System.out.println("Training : " + countTraining);
		System.out.println("Test : " + countTest);
		
		ALS als = new ALS();
		
		MatrixFactorizationModel model = als.setRank(1).setIterations(15).run(trainingRDD);
		
		
		Rating[] recommendProducts = model.recommendProducts(4169, 3);
		
		for(Rating rating : recommendProducts){
			System.out.println("Product : " + rating.product() + " Rating : " + rating.rating());
			
			
		}
		
		
		

		
		
		
	}
}
