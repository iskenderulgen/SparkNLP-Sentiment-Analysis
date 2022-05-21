/*Bu kod içerisinde reddit yorum csv dosyasındaki veriler satır satır okunarak analiz edilmiştir bu
 şekilde yorum bazında deta analiz yapılarak yüzde cinsinden sonuclara ulasmak mumkun olmustur.
 */
package MLlib;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

public class Naive_Bayes_Linebyline {
    public static int matematik_sayi = 0;
    public static int medical_sayi = 0;
    public static int sosyal_sayi = 0;
    public static int politik_sayi = 0;
    public static int male_sayi = 0;
    public static int female_sayi = 0;
    public static int spam_sayi = 0;
    public static int badword_sayi = 0;

    public static int spamteksayi = 0;
    public static int badwordteksayi = 0;



    public static final String DATA_PATH = "C:\\Users\\ULGEN\\Documents\\IdeaProjects\\Spark\\";


    public static void main(String[] args) {
        // Bir sparkcontex nesnesi tanımlanır
        JavaSparkContext javaSparkContext = new JavaSparkContext("local", "NaiveByesApp");
        // Bİr spark session nesnesi tanımlanır bu tanımlama ile json gibi veri formatları okunarak Dataset<row> nesnesi ıcıne atılabilir.
        SparkSession sparkSession = SparkSession.builder().master("local").appName("Naive_Bayes").getOrCreate();
        sparkSession.sparkContext().setLogLevel("ERROR");


        //keyword nesneleri javaRDD haline okunur
        JavaRDD<String> matematik = javaSparkContext.textFile(DATA_PATH + "Keywords2\\mathematical.txt");
        JavaRDD<String> medical = javaSparkContext.textFile(DATA_PATH + "Keywords2\\Medical.txt");
        JavaRDD<String> sosyal = javaSparkContext.textFile(DATA_PATH + "Keywords2\\Social.txt");
        JavaRDD<String> politik = javaSparkContext.textFile(DATA_PATH + "Keywords2\\politik.txt");

        JavaRDD<String> male = javaSparkContext.textFile(DATA_PATH + "Keywords2\\Bay_Bayan\\male.csv");
        JavaRDD<String> female = javaSparkContext.textFile(DATA_PATH + "Keywords2\\Bay_Bayan\\female.csv");

        JavaRDD<String> spam = javaSparkContext.textFile(DATA_PATH + "Keywords2\\spam\\spam.csv");
        JavaRDD<String> badword = javaSparkContext.textFile(DATA_PATH+"Keywords2\\badword\\badword.csv");
        JavaRDD<String> spam2 = javaSparkContext.textFile(DATA_PATH + "Keywords2\\spam\\spam.csv");
        JavaRDD<String> badword2 = javaSparkContext.textFile(DATA_PATH+"Keywords2\\badword\\badword.csv");

        final HashingTF tf = new HashingTF(20);

      /*  her bir keyword nesnesi /r/n olan yada hex olarak 0a0d olarak satır satır bölünür ve etiketlenme süreci baslar
           her keyword nesnesi uniqu ve belli bir kısıt olmadan itenilen rakamla etiketlenebilir.     */

        JavaRDD<LabeledPoint> matematikex = matematik.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(2.0,tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> medicalex = medical.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(3.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> sosyalex = sosyal.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(4.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> politikex = politik.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(5.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> maleex = male.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(6.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> femaleex = female.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(7.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> spamexp = spam.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(8.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> badwordex = badword.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(9.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> spamex2 = spam2.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("\r\n"); // kelimelere ayir
                return new LabeledPoint(10.0, tf.transform(Arrays.asList(keywords)));
            }
        });

        JavaRDD<LabeledPoint> badwordex2 = badword2.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] keywords = line.split("0a0d"); // kelimelere ayir
                return new LabeledPoint(11.0, tf.transform(Arrays.asList(keywords)));
            }
        });


        //etiketlenen keywordler tek nir nesne içerisinde birleştirilir.
        JavaRDD<LabeledPoint> analizlabel = medicalex.union(matematikex).union(politikex).union(sosyalex);
        // birleştirilen nesne navie- bayes modeli içerisinde eğitilir.
        NaiveBayesModel analizkeywordsmodel = NaiveBayes.train(analizlabel.rdd(),1.0);

        JavaRDD<LabeledPoint> baybayanlabel = maleex.union(femaleex);
        NaiveBayesModel baybayamodel = NaiveBayes.train(baybayanlabel.rdd(),1.0);

        JavaRDD<LabeledPoint> spambadwordlabel = spamexp.union(badwordex);
        NaiveBayesModel spambadwordmodel = NaiveBayes.train(spambadwordlabel.rdd(),1.0);


        NaiveBayesModel spammodel = NaiveBayes.train(spamex2.rdd(),1.0);
        NaiveBayesModel badwordmodel = NaiveBayes.train(badwordex2.rdd(),1.0);

        // analiz yapılacak olan 1M lik yorum dosyası onunarak dataset nesnesi içine atılır spark 2.0 dan sonra daha efektif şekilde kullanılmaktadır.
        //Dataset<Row> YorumDataset = sparkSession.read().text("C:\\Users\\ULGEN\\Documents\\IdeaProjects\\ApacheSpark\\data\\Yorum1M.csv");
        Dataset<Row> deneme = sparkSession.read().json(DATA_PATH+ "data\\2010haziran.json").drop("removal_reason","archived","author","author_flair_css_class",
                "author_flair_text","controversiality","created_utc","distinguished","downs","edited","gilded","id","link_id","name","parent_id","retrieved_on",
                "score","score_hidden","subreddit","subreddit_id","ups");
        deneme.createOrReplaceTempView("people");
        Dataset<Row> cleanset = sparkSession.sql("SELECT * FROM people  WHERE   body != '[A-Za-z0-9.,-]' AND   body !=  '[deleted]'  AND  body IS NOT NULL" );
        long total_yorum =cleanset.count();


        final long startTimefullk = System.currentTimeMillis();
        cleanset.foreach(kayit ->{
           String s = kayit.toString();
            Vector reddittest = tf.transform(Arrays.asList(s));
            double a = analizkeywordsmodel.predict(reddittest);
            double b = baybayamodel.predict(reddittest);
            double c = spambadwordmodel.predict(reddittest);
            double d = spammodel.predict(reddittest);
            double e = badwordmodel.predict(reddittest);
            //System.out.println("Toplu analiz="+a+"Bay bayan analiz="+b+"Spam analiz="+c);
            switch (((int) a)) {
                case 2:
                    matematik_sayi++;
                    break;
                case 3:
                    medical_sayi++;
                    break;
                case 4:
                    sosyal_sayi++;
                    break;
                case 5:
                    politik_sayi++;
                    break;
            }
            switch (((int) b)) {
                case 6:
                    male_sayi++;
                    break;
                case 7:
                    female_sayi++;
                    break;
            }
            switch (((int) c)){
                case 8:
                    spam_sayi++;
                    break;
                case 9:
                    badword_sayi++;
                    break;
            }

            switch (((int) d)){
                case 10:
                    spamteksayi++;
                    break;
            }
            switch (((int) e)){
                case 11:
                    badwordteksayi++;
                    break;
            }

        });
        final long endTimefullk = System.currentTimeMillis();
        System.out.println("Total  için geçen zaman: " + (endTimefullk - startTimefullk) );


        System.out.println("\n total yorum ="+total_yorum);

        System.out.println("\n matematik tahmini sayisi  = " + matematik_sayi +
                           "\n medikal tahmini sayisi = " + medical_sayi +
                           "\n sosyal tahmin sayisi   = " + sosyal_sayi +
                           "\n politik tahmin sayisi  = " + politik_sayi+
                           "\n\n male tahmin sayisi   = " + male_sayi +
                           "\n female tahmin sayisi   = " + female_sayi +
                           "\n\n spam tahmin sayisi   = " + spam_sayi +
                           "\n badword tahmin sayisi  = " + badword_sayi+
                           "\n\n spam  tek tahmin sayisi   = " + spamteksayi +
                           "\n\n badword tek  tahmin sayisi   = " + badwordteksayi);

        System.out.println("\n \n Matematik tahmin yüzdesi = %" +(double)matematik_sayi/total_yorum +
                            "\n medikal tahmin yüzdesi = %"     +(double)medical_sayi/total_yorum +
                            "\n sosyal tahmin yüzdesi  = %"     +(double)sosyal_sayi/total_yorum+
                            "\n politik tahmin yüzdesi = %"     +(double)politik_sayi/total_yorum+
                            "\n\n male tahmin yüzdesi  = %"     +(double)male_sayi/total_yorum+
                            "\n femalde tahmin yüzdesi = %"     +(double)female_sayi/total_yorum+
                            "\n\n spam tahmin yüzdesi  = %"     +(double)spam_sayi/total_yorum +
                            "\n badword tahmin yüzdesi ="       +(double)badword_sayi/total_yorum+
                            "\n\n spam tek tahmin yüzdesi  = %"     +(double)spamteksayi/total_yorum +
                            "\n\n badword tek tahmin yüzdesi  = %"     +(double)badwordteksayi/total_yorum ) ;
        System.out.println("/////////////////////////////////////////////////////////////////////////////////////////");

// bu blokta secilen miktarda analiz gerçekleştirilmektedir.
  /*      final long startTime1k = System.currentTimeMillis();
        secili_tahmin( 10000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime1k = System.currentTimeMillis();
        System.out.println("\n10K için geçen zaman: " + (endTime1k - startTime1k) );


        final long startTime2k = System.currentTimeMillis();
        secili_tahmin( 20000 , cleanset , tf , analizkeywordsmodel ,baybayamodel , spambadwordmodel  );
        final long endTime2k = System.currentTimeMillis();
        System.out.println("\n20K için geçen zaman: " + (endTime2k - startTime2k) );


        final long startTime5k = System.currentTimeMillis();
        secili_tahmin( 50000 , cleanset , tf , analizkeywordsmodel ,baybayamodel ,spambadwordmodel  );
        final long endTime5k = System.currentTimeMillis();
        System.out.println("\n50K için geçen zaman: " + (endTime5k - startTime5k) );


        final long startTime10k = System.currentTimeMillis();
        secili_tahmin( 100000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime10k = System.currentTimeMillis();
        System.out.println("\n100K için geçen zaman: " + (endTime10k - startTime10k) );


        final long startTime20k = System.currentTimeMillis();
        secili_tahmin( 200000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime20k = System.currentTimeMillis();
        System.out.println("\n200K için geçen zaman: " + (endTime20k - startTime20k) );


        final long startTime50k = System.currentTimeMillis();
        secili_tahmin( 500000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime50k = System.currentTimeMillis();
        System.out.println("\n500K için geçen zaman: " + (endTime50k - startTime50k) );


        final long startTime75k = System.currentTimeMillis();
        secili_tahmin( 750000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime75k = System.currentTimeMillis();
        System.out.println("\n750K için geçen zaman: " + (endTime75k - startTime75k) );


        final long startTime100k = System.currentTimeMillis();
        secili_tahmin( 1000000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime100k = System.currentTimeMillis();
        System.out.println("\n1M için geçen zaman: " + (endTime100k - startTime100k) );


        final long startTime150k = System.currentTimeMillis();
        secili_tahmin( 1500000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime150k = System.currentTimeMillis();
        System.out.println("\n1.5MK için geçen zaman: " + (endTime150k - startTime150k) );

        final long startTime300k = System.currentTimeMillis();
        secili_tahmin( 3000000 , cleanset , tf , analizkeywordsmodel , baybayamodel , spambadwordmodel  );
        final long endTime300k = System.currentTimeMillis();
        System.out.println("\n3MK için geçen zaman: " + (endTime300k - startTime300k) );*/

    }

   public static void secili_tahmin(int a, Dataset<Row> b, HashingTF c, NaiveBayesModel d, NaiveBayesModel h, NaiveBayesModel s) {

        int matematik_sayi2 = 0;
        int medical_sayi2   = 0;
        int sosyal_sayi2    = 0;
        int politik_sayi2   = 0;
        int male_sayi2      = 0;
        int female_sayi2    = 0;
        int spam_sayi2      = 0;
        int bad_word2       = 0;

        int i =0;
        List<Row> sayilirow = b.collectAsList();
            for (Row item :sayilirow ) {
                String s2 = item.toString();
                Vector reddittest2 = c.transform(Arrays.asList(s2));
                double a2 = d.predict(reddittest2);
                double b2 = h.predict(reddittest2);
                double c2 = s.predict(reddittest2);

                switch (((int) a2)) {
                    case 2:
                        matematik_sayi2++;
                        break;
                    case 3:
                        medical_sayi2++;
                        break;
                    case 4:
                        sosyal_sayi2++;
                        break;
                    case 5:
                        politik_sayi2++;
                        break;
                }

                switch (((int) b2)) {
                    case 6:
                        male_sayi2++;
                        break;
                    case 7:
                        female_sayi2++;
                }

                switch (((int) c2)) {
                    case 8:
                        spam_sayi2++;
                        break;
                    case 9:
                        bad_word2++;
                        break;
                }

                i++;
                if(i==a) break;
        }
        System.out.println("\n matematik tahmini sayisi  = " + matematik_sayi2 +
                "\n medikal tahmini sayisi = " + medical_sayi2 +
                "\n sosyal tahmin sayisi   = " + sosyal_sayi2 +
                "\n politik tahmin sayisi  = " + politik_sayi2 +
                "\n\n male tahmin sayisi   = " + male_sayi2 +
                "\n female tahmin sayisi   = " + female_sayi2 +
                "\n\n spam tahmin sayisi   = " + spam_sayi2 +
                "\n badword tahmin sayisi  = " + bad_word2);

        System.out.println("\n \n Matematik tahmin yüzdesi = %" +(double)matematik_sayi2/a +
                           "\n medikal tahmin yüzdesi = %"      +(double)medical_sayi2/a+
                           "\n politik tahmin yüzdesi = %"      +(double)politik_sayi2/a +
                           "\n\n male tahmin yüzdesi  = %"      +(double)male_sayi2/a +
                           "\n femalde tahmin yüzdesi = %"      +(double)female_sayi2/a+
                           "\n spam tahmin yüzdesi    = %"      +(double)spam_sayi2/a +
                           "\n badword tahmin yüzdesi ="        +(double)bad_word2/a +
                           "\n bakılan yorum sayisi =    "      + i) ;

        System.out.println("/////////////////////////////////////////////////////////////////////////////////////////");

    }

}
