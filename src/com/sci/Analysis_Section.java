package com.sci;

import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Collections;

class Analysis_Section  implements java.io.Serializable {
    // we create static values to store positive prediction values later to divide it total analyzied counts and get the
    // prediction results.
    private static int SVM_malecount =0, SVM_femalecount =0, SVM_spamcount =0, SVM_badwordcount =0, SVM_mathcount =0, SVM_medcount =0, SVM_socialcount =0, SVM_polcount =0;
    private static int NB_malecount =0, NB_femalecount =0, NB_spamcount =0, NB_badwordcount =0, NB_mathcount =0, NB_medcount =0, NB_socialcount =0, NB_polcount =0;
    private static int SVM_Malecount_New, SVM_Four_Union_Count,SVM_Spamcount_New,SVM_Math_Count_New,SVM_Pol_Count_New;

    static int[] SVM_Analysis(Dataset<Row> data, SVMModel model[], HashingTF tf){
        data.foreach((Row kayit) -> {
            //System.out.println(kayit.toString());
            Vector redditvector = tf.transform(Collections.singletonList(kayit.toString()));
            double bay   = model[0].predict(redditvector); if (bay == 1.0) SVM_malecount++;
            double bayan = model[1].predict(redditvector); if (bayan == 1.0) SVM_femalecount++;
            double spam  = model[2].predict(redditvector); if (spam == 1.0) SVM_spamcount++;
            double badword = model[3].predict(redditvector); if (badword == 1.0) SVM_badwordcount++;
            double math  = model[4].predict(redditvector); if(math == 1.0) SVM_mathcount++;
            double med   = model[5].predict(redditvector); if (med == 1.0) SVM_medcount++;
            double soc   = model[6].predict(redditvector); if(soc == 1.0) SVM_socialcount++;
            double pol   = model[7].predict(redditvector); if (pol== 1.0) SVM_polcount++;
        });
        int counters[] = {SVM_malecount,SVM_femalecount,SVM_spamcount,SVM_badwordcount,
                SVM_mathcount,SVM_medcount,SVM_socialcount,SVM_polcount};
        return counters;
    }

    static int[] SVM_Analysis_New(Dataset<Row> data, SVMModel model[], HashingTF tf){
        data.foreach((Row kayit) -> {
            Vector redditvector = tf.transform(Collections.singletonList(kayit.toString()));
            double bay   = model[0].predict(redditvector); if (bay == 1.0) SVM_Malecount_New++;
            double bay_four_union = model[1].predict(redditvector); if (bay_four_union == 1.0) SVM_Four_Union_Count++;
            double spam = model[2].predict(redditvector); if (spam == 1.0) SVM_Spamcount_New++;
            double math = model[3].predict(redditvector); if (math == 1.0) SVM_Math_Count_New++;
            double politic = model[4].predict(redditvector); if (politic == 1.0) SVM_Pol_Count_New++;
        });
        int counters[] = {SVM_Malecount_New,SVM_Four_Union_Count,SVM_Spamcount_New,SVM_Math_Count_New,SVM_Pol_Count_New};
        return counters;
    }

    static int[] NB_Analysis(Dataset<Row> data, NaiveBayesModel model[], HashingTF tf){
        data.foreach((Row kayit) -> {
            Vector redditvector = tf.transform(Collections.singletonList(kayit.toString()));
            double gender  = model[0].predict(redditvector);
            switch (((int)gender)){
                case 5:NB_malecount++;break;
                case 6:NB_femalecount++;break;}
            double spambadword = model[1].predict(redditvector);
            switch (((int)spambadword)) {
                case 7:NB_spamcount++;break;
                case 8:NB_badwordcount++;break;}
            double fourcat = model[2].predict(redditvector);
            switch (((int)fourcat)) {
                case 1:NB_mathcount++;break;
                case 2:NB_medcount++;break;
                case 3:NB_socialcount++;break;
                case 4:NB_polcount++;break;}
        });
        int counters[] = {NB_malecount,NB_femalecount,NB_spamcount,NB_badwordcount,
                NB_mathcount,NB_medcount,NB_socialcount,NB_polcount};
        return counters;
    }
}
