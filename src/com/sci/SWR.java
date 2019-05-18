/*
package com.sci;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class SWR implements java.io.Serializable{
     static String Stop_Words(SparkSession sparksession, String line) {

        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("raw")
                .setOutputCol("filtered");

        List<Row> data = Arrays.asList(
                RowFactory.create(Arrays.asList((line).split(" ")))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField(
                        "raw", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
        });

        Dataset<Row> dataset = sparksession.createDataFrame(data, schema);
        String SWRline = (remover.transform(dataset).drop("raw")).toString();
        return  SWRline;
    }
}


*/