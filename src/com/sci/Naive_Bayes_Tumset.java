/*Bu kod içerisinde reddit yorum csv dosyasındaki veriler tek bir RDD nesnesi içinde analiz edilmiştir */
package MLlib;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;


public class Naive_Bayes_Tumset {

    public static final String DATA_PATH = "C:\\Users\\ULGEN\\Documents\\IdeaProjects\\ApacheSpark\\Keywords\\";

    public static void main(String[] args) {

// Bir sparkcontex nesnesi tanımlanır
        JavaSparkContext javaSparkContext = new JavaSparkContext("local", "NaiveByesApp");

//keyword nesneleri javaRDD haline okunur
        JavaRDD<String> ateist = javaSparkContext.textFile(DATA_PATH + "Matematik_Ateist\\ateist.txt");
        JavaRDD<String> matematik = javaSparkContext.textFile(DATA_PATH + "Matematik_Ateist\\mathematical.txt");

        JavaRDD<String> medical = javaSparkContext.textFile(DATA_PATH + "Medical_Movie\\Medical.txt");
        JavaRDD<String> movie = javaSparkContext.textFile(DATA_PATH + "Medical_Movie\\Movie_words.txt");

        JavaRDD<String> Teen = javaSparkContext.textFile(DATA_PATH + "Sosyal_Gencler\\Teen.txt");
        JavaRDD<String> sosyal = javaSparkContext.textFile(DATA_PATH + "Sosyal_Gencler\\Social.txt");

        JavaRDD<String> politik = javaSparkContext.textFile(DATA_PATH + "Technologic_Politic\\politik.txt");
        JavaRDD<String> teknoloji = javaSparkContext.textFile(DATA_PATH + "Technologic_Politic\\technologic.txt");

        JavaRDD<String> male = javaSparkContext.textFile(DATA_PATH + "Bay_Bayan\\male.txt");
        JavaRDD<String> female = javaSparkContext.textFile(DATA_PATH + "Bay_Bayan\\female.txt");

        final HashingTF tf = new HashingTF(1000);

          /*  her bir keyword nesnesi   satır satır bölünür ve etiketlenme süreci baslar
           her keyword nesnesi uniqu ve belli bir kısıt olmadan itenilen rakamla etiketlenebilir.     */
        JavaRDD<LabeledPoint> ateizimex = ateist.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(1, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> matematikex = matematik.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(2, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> medicalex = medical.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(3, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> movieex = movie.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(4, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> teenex = Teen.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(5, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> sosyalex = sosyal.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(6, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> politikex = politik.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(7, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> teknolojiex = teknoloji.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(8, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> maleex = male.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(9, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> femaleex = female.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split(" "); // kelimelere ayir
                return new LabeledPoint(10, tf.transform(Arrays.asList(keywords)));
            }
        });

        //etiketlenen keywordler ikili olarak  nesne içerisinde birleştirilir. birleştirilen nesne navie- bayes modeli içerisinde eğitilir.

        JavaRDD<LabeledPoint> mat_ate = ateizimex.union(matematikex);
        NaiveBayesModel mat_ate_model  = NaiveBayes.train(mat_ate.rdd());

        JavaRDD<LabeledPoint> med_mov = medicalex.union(movieex);
        NaiveBayesModel med_mov_model  = NaiveBayes.train(med_mov.rdd());

        JavaRDD<LabeledPoint> teen_sos = teenex.union(sosyalex);
        NaiveBayesModel teen_sos_model  = NaiveBayes.train(teen_sos.rdd());

        JavaRDD<LabeledPoint> pol_tekno = politikex.union(teknolojiex);
        NaiveBayesModel pol_tekno_model  = NaiveBayes.train(pol_tekno.rdd());

        JavaRDD<LabeledPoint> male_femal = femaleex.union(maleex);
        NaiveBayesModel male_femal_model = NaiveBayes.train(male_femal.rdd());

        // analiz yapılacak olan 1M lik yorum dosyası okunarak JavaRDD nesnesi olarak tanımlanır.
        JavaRDD<String> reddit1 = javaSparkContext.textFile("C:\\Users\\ULGEN\\Documents\\IdeaProjects\\ApacheSpark\\data\\Yorum1M.csv");


        JavaRDD<String> redditRDD = reddit1.flatMap(new FlatMapFunction<String, String>() {
            public Iterator<String> call(String e) throws Exception {
                return Arrays.asList(e.split(" ")).iterator();
            }
        });


        // bu blokta secilen miktarda analiz gerçekleştirilmektedir.
        final long startTime1k = System.currentTimeMillis();
        spark ( 1000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime1k = System.currentTimeMillis();
        System.out.println("1K için geçen zaman: " + (endTime1k - startTime1k) );


        final long startTime2k = System.currentTimeMillis();
        spark ( 2000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime2k = System.currentTimeMillis();
        System.out.println("2K için geçen zaman: " + (endTime2k - startTime2k) );


        final long startTime5k = System.currentTimeMillis();
        spark ( 5000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime5k = System.currentTimeMillis();
        System.out.println("5K için geçen zaman: " + (endTime5k - startTime5k) );


        final long startTime10k = System.currentTimeMillis();
        spark ( 10000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime10k = System.currentTimeMillis();
        System.out.println("10K için geçen zaman: " + (endTime10k - startTime10k) );


        final long startTime20k = System.currentTimeMillis();
        spark ( 20000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime20k = System.currentTimeMillis();
        System.out.println("20K için geçen zaman: " + (endTime20k - startTime20k) );


        final long startTime50k = System.currentTimeMillis();
        spark ( 50000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime50k = System.currentTimeMillis();
        System.out.println("50K için geçen zaman: " + (endTime50k - startTime50k) );


        final long startTime75k = System.currentTimeMillis();
        spark ( 75000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime75k = System.currentTimeMillis();
        System.out.println("75K için geçen zaman: " + (endTime75k - startTime75k) );


        final long startTime100k = System.currentTimeMillis();
        spark ( 100000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime100k = System.currentTimeMillis();
        System.out.println("100K için geçen zaman: " + (endTime100k - startTime100k) );


        final long startTime150k = System.currentTimeMillis();
        spark ( 150000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime150k = System.currentTimeMillis();
        System.out.println("150K için geçen zaman: " + (endTime150k - startTime150k) );


        final long startTime1000k = System.currentTimeMillis();
        spark ( 1000000 , redditRDD , tf , mat_ate_model,med_mov_model,teen_sos_model, pol_tekno_model,male_femal_model);
        final long endTime1000k = System.currentTimeMillis();
        System.out.println("1000K için geçen zaman: " + (endTime1000k - startTime1000k) );

    }
    public static void spark (int a,
                              org.apache.spark.api.java.JavaRDD b,
                              org.apache.spark.mllib.feature.HashingTF c,
                              org.apache.spark.mllib.classification.NaiveBayesModel d,
                              org.apache.spark.mllib.classification.NaiveBayesModel e ,
                              org.apache.spark.mllib.classification.NaiveBayesModel f ,
                              org.apache.spark.mllib.classification.NaiveBayesModel g ,
                              org.apache.spark.mllib.classification.NaiveBayesModel h )
    {
        List<String> dizi = b.take(a);
        String reddit = dizi.toString();
        Vector reddittest = c.transform(Arrays.asList(reddit));
        System.out.println("Matematik içerikli tahmin : " + d.predict(reddittest));
        System.out.println("Medical içerikli tahmin : " + e.predict(reddittest));
        System.out.println("Sosyal  içerikli tahmin : " + f.predict(reddittest));
        System.out.println("politik içerikli tahmin : " + g.predict(reddittest));
        System.out.println("Bay & Bayan içerikli tahmin : " + h.predict(reddittest));
    }

}

