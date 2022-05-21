/*Bu kod içerisinde reddit yorum csv dosyasındaki veriler satır satır okunarak analiz edilmiştir bu
 şekilde yorum bazında deta analiz yapılarak yüzde cinsinden sonuclara ulasmak mumkun olmustur.
 Bu analiz sürecinde SVM Support Vector Machine yöntemi kullanılmıstır.
 */


package MLlib;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;



public class SVM_TekLabelKodu {

    // Veri dosyalarına ulasmak için ortak bir path dosyası olusuturulur.
    public static final String DATA_PATH = "C:\\Users\\ULGEN\\Documents\\IdeaProjects\\Spark\\";

    //yüzdelik şekilde hesaplama yaoaiblmek için her yorum dosaysının ait olduğu alan için counter nesneleri olusuturlmustur.
    public static int bayan = 0;
    public static int bay = 0;
    public static int spamcount = 0;
    public static int badwordcount = 0;
    public static int kategorisiz = 0;


    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf().set("spark.driver.memory","20g").setMaster("local").setAppName("SVMDeneme");
        JavaSparkContext javaSparkContext = new JavaSparkContext(conf);
        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config(conf)
                .getOrCreate();
       spark.sparkContext().setLogLevel("ERROR");



        Dataset<Row> deneme = spark.read().json(DATA_PATH+ "data\\2010haziran.json").drop("removal_reason","archived","author","author_flair_css_class","author_flair_text","controversiality","created_utc","distinguished","downs","edited","gilded","id","link_id","name","parent_id","retrieved_on","score","score_hidden","subreddit","subreddit_id","ups");
        deneme.createOrReplaceTempView("people");
        Dataset<Row> cleanset = spark.sql("SELECT * FROM people  WHERE   body != '[A-Za-z0-9.,-]' AND   body !=  '[deleted]'  AND  body IS NOT NULL" );

        int numIterations = 100;
        final HashingTF tf = new HashingTF(20);
        long total_yorum =cleanset.count();

//////////////////////////////////////////////////  Bay Model //////////////////////////////////////////////////////////
        final long startTimemale = System.currentTimeMillis();
        kategorisiz =0;

        JavaRDD<String> male = javaSparkContext.textFile(DATA_PATH + "Keywords2\\Bay_Bayan\\malecok.csv");

        JavaRDD<LabeledPoint> maleex = male.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split( "0a0d"); // kelimelere ayir
                return new LabeledPoint(1.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        final SVMModel baymodel = SVMWithSGD.train(maleex.rdd(), numIterations);
        baymodel.setThreshold(0.5);

        cleanset.foreach((Row kayit) -> {
            String s = kayit.toString();
            Vector reddittest = tf.transform(Arrays.asList(s));
            double a = baymodel.predict(reddittest);

            switch (((int) a)) {
                case 1:
                    bay++;
                    break;
                case 0:
                    kategorisiz++;
                    break;
            }
        });
        final long endTimemale = System.currentTimeMillis();
        System.out.println("\n Total  için geçen zaman   =  "  +(endTimemale - startTimemale) );
        System.out.println("\n total yorum               =  "  +total_yorum);
        System.out.println("\n bay tahmini sayisi        =  "  +bay);
        System.out.println("\n bay tahmin yüzdesi        =% "  +(double)bay/total_yorum );
        System.out.println("\n kategorisiz tahmini sayisi=  "  +kategorisiz);
        System.out.println("/////////////////////////////////////////////////////////////////////////////////////////");


////////////////////////////////////////////////  Bayan  Analizi  //////////////////////////////////////////////////////
        final long startTimefemale = System.currentTimeMillis();
        kategorisiz=0;

        JavaRDD<String> female = javaSparkContext.textFile(DATA_PATH + "Keywords2\\Bay_Bayan\\femalecok2.csv");

        JavaRDD<LabeledPoint> femaleex = female.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("0a0d"); // kelimelere ayir
                return new LabeledPoint(1.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        final SVMModel femalemodel = SVMWithSGD.train(femaleex.rdd(), numIterations);
        femalemodel.setThreshold(0.5);


        cleanset.foreach((Row kayit) -> {
            String s = kayit.toString();
            Vector reddittest = tf.transform(Arrays.asList(s));
            double a = femalemodel.predict(reddittest);

            switch (((int) a)) {
                case 1:
                    bayan++;
                    break;
                case 0:
                    kategorisiz++;
                    break;
            }
        });
        final long endTimefemale = System.currentTimeMillis();
        System.out.println("\n Total  için geçen zaman   =  "  +(endTimefemale - startTimefemale) );
        System.out.println("\n total yorum               =  "  +total_yorum);
        System.out.println("\n bayan tahmini sayisi      =  "  +bayan);
        System.out.println("\n bayan tahmin yüzdesi      =% "  +(double)bayan/total_yorum );
        System.out.println("\n kategorisiz tahmini sayisi=  "  +kategorisiz);
        System.out.println("/////////////////////////////////////////////////////////////////////////////////////////");

/////////////////////////////////////////////////////////SPAM ANALİZİ///////////////////////////////////////////////////

        JavaRDD<String> spam = javaSparkContext.textFile(DATA_PATH + "Keywords2\\spam\\spam.csv");

        JavaRDD<LabeledPoint> spamex = spam.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("0a0d"); // kelimelere ayir
                return new LabeledPoint(1.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        final SVMModel spammodel = SVMWithSGD.train(spamex.rdd(), numIterations);
        spammodel.setThreshold(0.5);

        cleanset.foreach((Row spamkayit)->{
            String s = spamkayit.toString();
            Vector reddittest = tf.transform(Arrays.asList(s));
            double a = spammodel.predict(reddittest);
            if(a == 1.0){
                spamcount++;
            }
        });

        System.out.println("\n spam tahmini sayisi = " +spamcount);
        System.out.println("\n spam tahmin yüzdesi = %" +(double)spamcount/total_yorum);
        System.out.println("/////////////////////////////////////////////////////////////////////////////////////////");

////////////////////////////////////////////////////////Küfür Analizi///////////////////////////////////////////////////

        JavaRDD<String> badword = javaSparkContext.textFile(DATA_PATH + "Keywords2\\badword\\badword.csv");

        JavaRDD<LabeledPoint> badwordex = badword.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("0a0d"); // kelimelere ayir
                return new LabeledPoint(1.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        final SVMModel badwordmodel = SVMWithSGD.train(badwordex.rdd(), numIterations);
        badwordmodel.setThreshold(0.5);
        cleanset.foreach((Row badwordkayit)->{
            String s = badwordkayit.toString();
            Vector reddittest = tf.transform(Arrays.asList(s));
            double a = badwordmodel.predict(reddittest);
            if(a == 1.0){
                badwordcount++;
            }
        });

        System.out.println("\n badword tahmini sayisi =  " +badwordcount);
        System.out.println("\n badword tahmin yüzdesi = %" +(double)badwordcount/total_yorum);
        System.out.println("/////////////////////////////////////////////////////////////////////////////////////////");

//////////////////////////////////////////////////////// Secilen sete gore tahmin //////////////////////////////////////


        final long startTime1k = System.currentTimeMillis();
        secili_tahmin( 10000 , cleanset , tf , spammodel );
        final long endTime1k = System.currentTimeMillis();
        System.out.println("\n 10K için geçen zaman: " + (endTime1k - startTime1k) );


        final long startTime2k = System.currentTimeMillis();
        secili_tahmin( 20000 , cleanset , tf , spammodel );
        final long endTime2k = System.currentTimeMillis();
        System.out.println("\n 20K için geçen zaman: " + (endTime2k - startTime2k) );


        final long startTime5k = System.currentTimeMillis();
        secili_tahmin( 50000 , cleanset , tf , spammodel );
        final long endTime5k = System.currentTimeMillis();
        System.out.println("\n 50K için geçen zaman: " + (endTime5k - startTime5k) );


        final long startTime10k = System.currentTimeMillis();
        secili_tahmin( 100000 , cleanset , tf , spammodel );
        final long endTime10k = System.currentTimeMillis();
        System.out.println("\n 100K için geçen zaman: " + (endTime10k - startTime10k) );


        final long startTime20k = System.currentTimeMillis();
        secili_tahmin( 200000 , cleanset , tf , spammodel );
        final long endTime20k = System.currentTimeMillis();
        System.out.println("\n 200K için geçen zaman: " + (endTime20k - startTime20k) );


        final long startTime50k = System.currentTimeMillis();
        secili_tahmin( 500000 , cleanset , tf , spammodel );
        final long endTime50k = System.currentTimeMillis();
        System.out.println("\n 500K için geçen zaman: " + (endTime50k - startTime50k) );


        final long startTime75k = System.currentTimeMillis();
        secili_tahmin( 750000 , cleanset , tf , spammodel );
        final long endTime75k = System.currentTimeMillis();
        System.out.println("\n 750K için geçen zaman: " + (endTime75k - startTime75k) );


        final long startTime100k = System.currentTimeMillis();
        secili_tahmin( 1000000 , cleanset , tf , spammodel );
        final long endTime100k = System.currentTimeMillis();
        System.out.println("\n 1M için geçen zaman: " + (endTime100k - startTime100k) );


        final long startTime150k = System.currentTimeMillis();
        secili_tahmin( 1500000 , cleanset , tf , spammodel);
        final long endTime150k = System.currentTimeMillis();
        System.out.println("\n 1.5M için geçen zaman: " + (endTime150k - startTime150k) );


        final long startTime300k = System.currentTimeMillis();
        secili_tahmin( 3000000 , cleanset , tf , spammodel );
        final long endTime300k = System.currentTimeMillis();
        System.out.println("\n 3M için geçen zaman: " + (endTime300k - startTime300k) );

    }




    public static void secili_tahmin(int bakılacaksayi, Dataset<Row> seciliset, HashingTF c,  SVMModel h ) {

        int i =0;
        int spam2 =0;

        List<Row> sayilirow = seciliset.collectAsList();
        for (Row item :sayilirow ) {
            String s2 = item.toString();
            Vector reddittest2 = c.transform(Arrays.asList(s2));
            double a2 = h.predict(reddittest2);
            if(a2 == 1){
                spam2++;
            }
            i++;
            if(i==bakılacaksayi) break;
        }

        System.out.println("\n spam tahmini sayisi = " + spam2);
        System.out.println("\n spam tahmin yüzdesi  = %"   +(double) spam2 /bakılacaksayi+
                "\n bakılan yorum sayisi ="     + i) ;
        System.out.println("/////////////////////////////////////////////////////////////////////////////////////////");
    }
}




/*  seciliset.foreach(new ForeachFunction<Row>() {

            @Override
            public void call(Row kayit) throws Exception {
                String s = kayit.toString();
                Vector reddittest = c.transform(Arrays.asList(s));
                double a = h.predict(reddittest);
                if (a == 0.0) {
                    bay++;
                }
                if (a == 1.0) {
                    bayan++;
                }
                i++;
            }
        });
*/