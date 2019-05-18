package com.sci;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import javax.jws.WebParam;


public class Text_Analysis {

    // we determine the datapaths of dataset and keywords for ease of use in future operation.
    private static final String DATA_PATH = "C:\\Users\\ULGEN\\Documents\\IdeaWorkspace\\Reddit_Analysis\\";
    private static final String DATASET_PATH = DATA_PATH+"2007.json";
    private static final String MALE_PATH = DATA_PATH+"Keywords2/Bay_Bayan/malecok.csv";
    private static final String FEMALE_PATH = DATA_PATH+"Keywords2/Bay_Bayan/femalecok2.csv";
    private static final String SPAM_PATH = DATA_PATH+"Keywords2/spam/spam.csv";
    private static final String BADWORD_PATH = DATA_PATH+"Keywords2/badword/badword.csv";
    private static final String MATHEMATIC_PATH = DATA_PATH+"Keywords2/mathematical.csv";
    private static final String MEDICAL_PATH = DATA_PATH+"Keywords2/Medical.csv";
    private static final String SOCIAL_PATH = DATA_PATH+"Keywords2/Social.csv";
    private static final String POLITICS_PATH = DATA_PATH+"Keywords2/politik.csv";


    public  static void main (String[] args){
        //We create one conf prameter to use it on boths spark session and spark context.
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Text_Mining_App");
        JavaSparkContext javasparkcontext = new JavaSparkContext(conf);
        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config(conf)
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

              /* in this section we read data by cleaning. we exclude uncesserray columns and read only "body" column
              also we exluce deleted comments and comments that has long URL lines.*/

        final long Dataclean_begin_clock = System.currentTimeMillis();
        Dataset<Row> reddit = spark.read()
                .json(DATASET_PATH)
                .select("body")
                .where("body !=  '[deleted]' AND body NOT LIKE '%www.%' AND body NOT LIKE '%http%'")
                .filter("body IS NOT NULL");
        final long Dataclean_end_clock = System.currentTimeMillis();
        reddit.printSchema();
        reddit.show();
        Long data_time = Dataclean_end_clock-Dataclean_begin_clock;
        System.out.println("Total Time For Data Clean and Integration=\t"+data_time+"  Milisec\t"+data_time/1000+"  Seconds\t"+data_time/60000+"  Minutes");
        long total_comments = reddit.count();
        System.out.println("Total Comment Lines To Be Analized ="+total_comments+"\n\n\n");

        //We determine TF value and iteration values to use on multiple classes.
        final HashingTF tf = new HashingTF(20);
        int iteration = 100;
        double SVM_Label = 1.0;

        // SVM Analysis Section Starts Here
        double Male_Precision = Model_Evuluation.SVM_Precision(iteration,MALE_PATH,SVM_Label,javasparkcontext,tf);
        double Female_Precision = Model_Evuluation.SVM_Precision(iteration,FEMALE_PATH,SVM_Label,javasparkcontext,tf);
        double Spam_Precision = Model_Evuluation.SVM_Precision(iteration,SPAM_PATH,SVM_Label,javasparkcontext,tf);
        double Badword_Precision = Model_Evuluation.SVM_Precision(iteration,BADWORD_PATH,SVM_Label,javasparkcontext,tf);
        double Math_Precision = Model_Evuluation.SVM_Precision(iteration,MATHEMATIC_PATH,SVM_Label,javasparkcontext,tf);
        double Med_Precision = Model_Evuluation.SVM_Precision(iteration,MEDICAL_PATH,SVM_Label,javasparkcontext,tf);
        double Soc_Precision = Model_Evuluation.SVM_Precision(iteration,SOCIAL_PATH,SVM_Label,javasparkcontext,tf);
        double Pol_Precision = Model_Evuluation.SVM_Precision(iteration,POLITICS_PATH,SVM_Label,javasparkcontext,tf);

        System.out.println("*********SUPPORT VECTOR MACHINE MODEL PRECISION ANALYSIS INITIATING*********\n");
        System.out.println("Male Model SVM Precision is     =\t%"+Male_Precision);
        System.out.println("Female Model SVM Precision is   =\t%"+Female_Precision);
        System.out.println("Spam Model SVM Precision is     =\t%"+Spam_Precision);
        System.out.println("Bad Word Model SVM Precision is =\t%"+Badword_Precision);
        System.out.println("Mathematical Model SVM Precision is=\t%"+Math_Precision);
        System.out.println("Medical Model SVM Precision is  =\t%"+Med_Precision);
        System.out.println("Social Model SVM Precision is   =\t%"+Soc_Precision);
        System.out.println("Political Model SVM Precision is=\t%"+Pol_Precision);
        System.out.println("\n*********SUPPORT VECTOR MACHINE MODEL PRECISION ANALYSIS COMPELETED*********\n\n\n\n\n");

        SVMModel malemodel = ML_Models.SVM_Model(iteration, MALE_PATH,SVM_Label,javasparkcontext,tf);
        SVMModel femalemodel = ML_Models.SVM_Model(iteration, FEMALE_PATH,SVM_Label,javasparkcontext,tf);
        SVMModel spammodel = ML_Models. SVM_Model(iteration,SPAM_PATH,SVM_Label,javasparkcontext,tf);
        SVMModel badwordmodel = ML_Models.SVM_Model(iteration,BADWORD_PATH,SVM_Label,javasparkcontext,tf);
        SVMModel mathmodel = ML_Models.SVM_Model(iteration,MATHEMATIC_PATH,SVM_Label,javasparkcontext,tf);
        SVMModel medicmodel = ML_Models.SVM_Model(iteration,MEDICAL_PATH,SVM_Label,javasparkcontext,tf);
        SVMModel socialmodel = ML_Models.SVM_Model(iteration,SOCIAL_PATH,SVM_Label,javasparkcontext,tf);
        SVMModel polmodel = ML_Models.SVM_Model(iteration,POLITICS_PATH,SVM_Label,javasparkcontext,tf);


        // in this section Support Vector Machine Algoritm makes the categorization using Analysis_Section class.
        SVMModel SVM_dizi_model [] = {malemodel,femalemodel,spammodel,badwordmodel,mathmodel,medicmodel,socialmodel,polmodel};
        final long SVM_begin_clock = System.currentTimeMillis();
        int sonuc[] = Analysis_Section.SVM_Analysis(reddit,SVM_dizi_model,tf);
        final long SVM_end_clock = System.currentTimeMillis();
        long SVM_time = SVM_end_clock-SVM_begin_clock;

        System.out.println("*********SUPPORT VECTOR MACHINE ANALYSIS INITIATING*********\n");
        System.out.println("Time elapsed for Support Vector Machine Categorization =  "+SVM_time+"  Milisec\t"+(SVM_time/1000.0)+"  Seconds\t"+(SVM_time/60000)+"  Minutes");
        System.out.println("\nMale Probability Count      =\t"+ sonuc[0] +"\tMale Probability Percent      =\t%"+100*((double) sonuc[0] /total_comments));
        System.out.println("Female Probability Count    =\t"+ sonuc[1] +"\tFemale Probability Percent    =\t%"+100*((double) sonuc[1] /total_comments));
        System.out.println("Spam Probability Count      =\t"+ sonuc[2] +"\tSpam Probability Percent        =\t%"+100*((double) sonuc[2] /total_comments));
        System.out.println("Badword Probability Count   =\t"+ sonuc[3] +"\tBadword Probability Percent  =\t%"+100*((double) sonuc[3] /total_comments));
        System.out.println("Mathematical Probability Count=\t"+ sonuc[4] +"\tMathematical Probability Percent=\t%"+100*((double) sonuc[4] /total_comments));
        System.out.println("Medical Probability Count   =\t"+ sonuc[5] +"\tMedical Probability Percent      =\t%"+100*((double) sonuc[5] /total_comments));
        System.out.println("Social Probability Count    =\t"+ sonuc[6] +"\tSocial Probability Percent=\t%"+100*((double) sonuc[6] /total_comments));
        System.out.println("Politics Probability Count  =\t"+ sonuc[7] +"\tPolitics Probability Percent =\t%"+100*((double) sonuc[7] /total_comments));
        System.out.println("\n*********SUPPORT VECTOR MACHINE ANALYSIS COMPLETED*********\n\n\n\n");






        //SVM TWO CLASS UNIONIZED ANALYSIS SECTION
        //Two Class SVM Model 1.0 for desired features 0.0 for non desired features./ İki Class SVM Model dogrulama ve Analiz Aşaması Buradadır.
        JavaRDD<LabeledPoint> Male_Label_New = ML_Models.labelingdata(MALE_PATH,1.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Female_Label_New = ML_Models.labelingdata(FEMALE_PATH,0.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Male_Female_New_Label = Male_Label_New.union(Female_Label_New);

        JavaRDD<LabeledPoint> Male_Label_New_2 = ML_Models.labelingdata(MALE_PATH,1.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Female_Label_New_2 = ML_Models.labelingdata(FEMALE_PATH,0.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Badword_New_Label_2 = ML_Models.labelingdata(BADWORD_PATH,0.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Male_Female_New_FOur_Union_Label = Male_Label_New_2.union(Female_Label_New_2).union(Badword_New_Label_2);

        JavaRDD<LabeledPoint> Spam_New_Label = ML_Models.labelingdata(SPAM_PATH,1.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Badword_New_Label = ML_Models.labelingdata(BADWORD_PATH,0.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Spam_Badword_New_Label = Spam_New_Label.union(Badword_New_Label);

        JavaRDD<LabeledPoint> Math_Label_New = ML_Models.labelingdata(MATHEMATIC_PATH,1.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Medic_Label_new = ML_Models.labelingdata(MEDICAL_PATH,0.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Math_Medic_New_Label = Math_Label_New.union(Medic_Label_new);

        JavaRDD<LabeledPoint> Social_Label_New = ML_Models.labelingdata(SOCIAL_PATH,1.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Politics_Label_New = ML_Models.labelingdata(POLITICS_PATH,0.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> Social_Politics_New_Label = Social_Label_New.union(Politics_Label_New);
        
        double Male_New_Svm_Precision = Model_Evuluation.SVM_Precision_New(iteration,Male_Female_New_Label);
        double Male_Four_Union_Precision = Model_Evuluation.SVM_Precision_New(iteration,Male_Female_New_FOur_Union_Label);
        double Spam_New_Svm_Precision = Model_Evuluation.SVM_Precision_New(iteration,Spam_Badword_New_Label);
        double Math_Medic_New_Precision = Model_Evuluation.SVM_Precision_New(iteration,Math_Medic_New_Label);
        double Social_Politics_New_Precision = Model_Evuluation.SVM_Precision_New(iteration,Social_Politics_New_Label);


        System.out.println("*********UNIONIZED NEW SUPPORT VECTOR MACHINE MODEL PRECISION ANALYSIS INITIATING*********\n");
        System.out.println("Male Model SVM NEW Precision is     =\t%"+Male_New_Svm_Precision);
        System.out.println("Male Model Four Union SVM NEW Precision is=\t%"+Male_Four_Union_Precision);
        System.out.println("Spam Model SVM NEW Precision is     =\t%"+Spam_New_Svm_Precision);
        System.out.println("Math Model SVM NEW Precision is     =\t%"+Math_Medic_New_Precision);
        System.out.println("Social Model SVM NEW Precision is   =\t%"+Social_Politics_New_Precision);
        System.out.println("\n*********UNIONIZED NEW SUPPORT VECTOR MACHINE MODEL PRECISION ANALYSIS COMPELETED*********\n\n\n\n\n");

        SVMModel Male_Female_New_Model = ML_Models.SVM_Model_2(iteration,Male_Female_New_Label);
        SVMModel Male_Female_Four_Union_Model = ML_Models.SVM_Model_2(iteration,Male_Female_New_FOur_Union_Label);
        SVMModel Spam_Badword_New_Model = ML_Models.SVM_Model_2(iteration,Spam_Badword_New_Label);
        SVMModel Math_Medic_New_Model = ML_Models.SVM_Model_2(iteration,Math_Medic_New_Label);
        SVMModel Social_Politics_New_Model = ML_Models.SVM_Model_2(iteration,Social_Politics_New_Label);


        SVMModel SVM_dizi_New_model [] = {Male_Female_New_Model,Male_Female_Four_Union_Model,Spam_Badword_New_Model,Math_Medic_New_Model,Social_Politics_New_Model};
        final long SVM2_begin_clock = System.currentTimeMillis();
        int sonuc_new[] = Analysis_Section.SVM_Analysis_New(reddit,SVM_dizi_New_model,tf);
        final long SVM2_end_clock = System.currentTimeMillis();
        long SVM2_time = SVM2_end_clock-SVM2_begin_clock;

        System.out.println("*********UNIONIZED SUPPORT VECTOR MACHINE NEW ANALYSIS INITIATING*********\n");
        System.out.println("Time elapsed for Support Vector Machine Categorization =  "+SVM2_time+"  Milisec\t"+(SVM2_time/1000.0)+"  Seconds\t"+(SVM2_time/60000)+"  Minutes");
        System.out.println("\nMale Probability NEW Count         =\t"+ sonuc_new[0] +"\tMale Probability NEW Percent           =\t%"+100*((double) sonuc_new[0] /total_comments));
        System.out.println("Male Four Union Probability NEW Count=\t"+ sonuc_new[1] +"\tMale Four Union Probability NEW Percent=\t%"+100*((double) sonuc_new[1] /total_comments));
        System.out.println("Spam Probability NEW Count           =\t"+ sonuc_new[2] +"\tFemale Probability NEW Percent         =\t%"+100*((double) sonuc_new[2] /total_comments));
        System.out.println("Mathematics Probability NEW Count    =\t"+ sonuc_new[3] +"\tMathematics Probability NEW Percent    =\t%"+100*((double) sonuc_new[3] /total_comments));
        System.out.println("Politcs Probability NEW Count        =\t"+ sonuc_new[4] +"\tPolitics Probability NEW Percent       =\t%"+100*((double) sonuc_new[4] /total_comments));
        System.out.println("\n*********UNIONIZED SUPPORT VECTOR MACHINE NEW ANALYSIS COMPLETED*********\n\n\n\n");
        //SVM TWO CLASS UNIONIZED ANALYSIS SECTION
        // SVM Analysis Section Ends Here








        //Naive Bayes Analysis Starts Here
        JavaRDD<LabeledPoint> labeledmath = ML_Models.labelingdata(MATHEMATIC_PATH,1.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeledmedic = ML_Models.labelingdata(MEDICAL_PATH,2.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeledsocial = ML_Models.labelingdata(SOCIAL_PATH,3.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeledpol = ML_Models.labelingdata(POLITICS_PATH,4.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeled4cat = labeledmath.union(labeledmedic).union(labeledsocial).union(labeledpol);

        JavaRDD<LabeledPoint> labeledmale = ML_Models.labelingdata(MALE_PATH,5.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeledfemale = ML_Models.labelingdata(FEMALE_PATH,6.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeledgender = labeledmale.union(labeledfemale);

        JavaRDD<LabeledPoint> labeledspam = ML_Models.labelingdata(SPAM_PATH,7.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeledbadword = ML_Models.labelingdata(BADWORD_PATH,8.0,javasparkcontext,tf);
        JavaRDD<LabeledPoint> labeledspambadword = labeledspam.union(labeledbadword);

        double lambda = 1.0;
        // Model evaluation analysis made here using model evaluation class.
        double male_accuaricy   = Model_Evuluation.NB_Accuaricy(MALE_PATH,1.0,javasparkcontext,tf,lambda);
        double female_accuaricy = Model_Evuluation.NB_Accuaricy(FEMALE_PATH,1.0,javasparkcontext,tf,lambda);
        double spam_accuaricy   = Model_Evuluation.NB_Accuaricy(SPAM_PATH,1.0,javasparkcontext,tf,lambda);
        double badword_accuaricy = Model_Evuluation.NB_Accuaricy(BADWORD_PATH,1.0,javasparkcontext,tf,lambda);
        double math_accuaricy   = Model_Evuluation.NB_Accuaricy(MATHEMATIC_PATH,1.0,javasparkcontext,tf,lambda);
        double med_accuaricy    = Model_Evuluation.NB_Accuaricy(MEDICAL_PATH,1.0,javasparkcontext,tf,lambda);
        double social_accuaricy = Model_Evuluation.NB_Accuaricy(SOCIAL_PATH,1.0,javasparkcontext,tf,lambda);
        double pol_accuaricy    = Model_Evuluation.NB_Accuaricy(POLITICS_PATH,1.0,javasparkcontext,tf,lambda);


        //Burada birleştirilmiş labeledpoint verisi NB doğrulama koduna gönderiliyoru %100
        //dönüş verisinden kaçınmak için denenmiş bir bypass
        System.out.println("*********NAİVE BAYES UNIONIZED MODEL PRECISION ANALYSIS INITIATING*********\n");
        double male_accuaricy_2   = Model_Evuluation.NB_Accuaricy_bypass(labeledgender,lambda);
        double spam_accuaricy_2   = Model_Evuluation.NB_Accuaricy_bypass(labeledspambadword,lambda);
        double fourcat_accuaricy_2   = Model_Evuluation.NB_Accuaricy_bypass(labeled4cat,lambda);
        System.out.println("Male & Female Union Model Naive Bayes Precision is=\t%"+male_accuaricy_2);
        System.out.println("Spam & Badword Union Model Naive Bayes Precision is=\t%"+spam_accuaricy_2);
        System.out.println("Four category Union Model Naive Bayes Precision is=\t%"+fourcat_accuaricy_2+"\n");
        System.out.println("*********NAİVE BAYES UNIONIZED MODEL PRECISION ANALYSIS COMPLETED*********\n\n\n\n");



        System.out.println("*********NAİVE BAYES MODEL PRECISION ANALYSIS INITIATING*********\n");
        System.out.println("Male Model Naive Bayes Precision is     =\t%"+male_accuaricy);
        System.out.println("Female Model Naive Bayes Precision is   =\t%"+female_accuaricy);
        System.out.println("Spam Model Naive Bayes Precision is     =\t%"+spam_accuaricy);
        System.out.println("Bad Word Model Naive Bayes Precision is =\t%"+badword_accuaricy);
        System.out.println("Mathematical Naive Bayes SVM Precision is=\t%"+math_accuaricy);
        System.out.println("Medical Model Naive Bayes Precision is  =\t%"+med_accuaricy);
        System.out.println("Social Model Naive Bayes Precision is   =\t%"+social_accuaricy);
        System.out.println("Political Model Naive Bayes Precision is=\t%"+pol_accuaricy);
        System.out.println("\n*********NAİVE BAYES MODEL PRECISION ANALYSIS COMPLETED*********\n\n\n\n\n");

        //we unionize labeled datas NB if we use each model for analysis model predicts always returns same value.
        NaiveBayesModel gendermodel  = ML_Models.NB_Model(labeledgender,lambda);
        NaiveBayesModel spambadwordmodel = ML_Models.NB_Model(labeledspambadword,lambda);
        NaiveBayesModel fourcatmodel = ML_Models.NB_Model(labeled4cat,lambda);

        NaiveBayesModel NB_dizi_model [] = {gendermodel,spambadwordmodel,fourcatmodel};
        final long NB_begin_clock = System.currentTimeMillis();
        int NB_sonuc[] = Analysis_Section.NB_Analysis(reddit,NB_dizi_model,tf);
        final long NB_end_clock = System.currentTimeMillis();
        long NB_time = NB_end_clock - NB_begin_clock;

        System.out.println("*********NAİVE BAYES  ANALYSIS INITIATING*********\n");
        System.out.println("Time elapsed for Naive Bayes Categorization =  "+NB_time+"  Milisec\t"+(NB_time/1000)+"  Seconds\t"+NB_time/60000+"  Minutes");
        System.out.println("\nMale Probability Count      =\t"+ NB_sonuc[0] +"\tMale Probability Percent         =\t%"+ 100*((double) NB_sonuc[0] /total_comments));
        System.out.println("Female Probability Count    =\t"+ NB_sonuc[1] +"\tFemale Probability Percent     =\t%"+ 100*((double) NB_sonuc[1] /total_comments));
        System.out.println("Spam Probability Count      =\t"+ NB_sonuc[2] +"\tSpam Probability Percent         =\t%"+100*((double) NB_sonuc[2] /total_comments));
        System.out.println("Badword Probability Count   =\t"+ NB_sonuc[3] +"\tBadword Probability Percent   =\t%"+100*((double) NB_sonuc[3] /total_comments));
        System.out.println("Mathematical Probability Count=\t"+ NB_sonuc[4] +"\tMathematical Probability Percent=\t%"+100*((double) NB_sonuc[4] /total_comments));
        System.out.println("Medical Probability Count   =\t"+ NB_sonuc[5] +"\tMedical Probability Percent       =\t%"+100*((double) NB_sonuc[5] /total_comments));
        System.out.println("Social Probability Count    =\t"+ NB_sonuc[6] +"\tMathematical Probability Percent=\t%"+100*((double) NB_sonuc[6] /total_comments));
        System.out.println("Politics Probability Count  =\t"+ NB_sonuc[7] +"\tMathematical Probability Percent  =\t%"+100*((double) NB_sonuc[7] /total_comments));
        System.out.println("\n*********NAİVE BAYES ANALYSIS COMPLETED*********\n\n\n\n\n");
        //Naive Bayes Analysis Ends Here
    }
}
