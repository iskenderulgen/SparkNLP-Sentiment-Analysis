# Sentiment Analysis with Apache Spark NLP

This project demonstrates text classification capabilities using Apache Spark and John Snow Labs' Spark NLP library. It implements sentiment analysis on movie reviews using pre-trained GloVe word embeddings and logistic regression.

## Overview

The project showcases a modern approach to text classification by using:
- Pre-trained GloVe word embeddings (100-dimensional)
- Spark NLP pipeline for text processing
- Logistic Regression for classification
- Applying trained model on unseed dataset

## Architecture

The pipeline follows these steps:
1. Text preprocessing and cleaning
2. Tokenization
3. Word embedding generation using GloVe
4. Sentence embedding creation (averaging word embeddings)
5. Model training using Logistic Regression
6. Model evaluation
7. Application to unseen data

## Requirements

- Python 3.11+
- PySpark 3.3.1
- Spark NLP 5.5.3+
- JDK 8

## Datasets

The project uses two datasets from Hugging Face:
- [Stanford IMDB Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) - Used for training the model
- [Yelp Reviews](https://huggingface.co/datasets/Yelp/yelp_review_full) - Used for testing transfer learning capabilities

## Pre-trained Models

- [Glove Word Embeddings](https://sparknlp.org/2020/01/22/glove_100d.html) - GloVe word embeddings (100-dimensional) loaded locally from the data directory

## Setup and Execution

1. Clone the repository
2. Download the required datasets from Hugging Face and place them in the `data/` directory
3. Download GloVe embeddings for Spark NLP and extract to `data/glove_100d/`
4. Run the Jupyter notebook `main.ipynb`

## Implementation Details

### 1. Spark Session Configuration

```python
spark = (
    SparkSession.builder.appName("Spark-Text-Classification")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max", "2000M")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3")
    .getOrCreate()
)
```
Spark-NLP jar file downloads large amount of supporting jar files so process might take a while.

### 2. NLP Pipeline

The project uses a comprehensive pipeline for text processing:
- `DocumentAssembler` - Prepares raw text for NLP
- `Tokenizer` - Breaks text into tokens
- `WordEmbeddingsModel` - Applies GloVe embeddings
- `SentenceEmbeddings` - Creates document-level embeddings
- `EmbeddingsFinisher` - Converts embeddings to feature vectors

### 3. Model Training

The model uses Spark's LogisticRegression with the following:
- Features from the NLP pipeline
- Binary classification for sentiment analysis
- 80/20 train/test split

### 4. Transfer Learning

The trained model is then applied to a completely different dataset (Yelp reviews) to demonstrate its generalization capabilities.

## Results

The model achieves good classification performance on the IMDB dataset and shows effective transfer learning capabilities when applied to the Yelp reviews dataset.

## Benefits of this Approach

- **Better NLP Techniques**: Uses word embeddings instead of traditional TF-IDF or Count Vectorizers
- **Scalability**: Built on Apache Spark for handling large datasets
- **Production-Ready**: The pipeline can be deployed in a production environment

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## References

- [Spark NLP Documentation](https://nlp.johnsnowlabs.com/)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [IMDB Dataset on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Yelp Reviews Dataset on Hugging Face](https://huggingface.co/datasets/Yelp/yelp_review_full)

This work is a modernized and simplified version of the following research 

```
@inproceedings{7960721,
  author={Oğul, İskender Ülgen and Özcan, Caner and Hakdağlı, Özlem},
  booktitle={2017 25th Signal Processing and Communications Applications Conference (SIU)}, 
  title={Fast text classification with Naive Bayes method on Apache Spark}, 
  year={2017},
  volume={},
  number={},
  pages={1-4},
  keywords={Sparks;Java;Internet of Things;Standards;Text categorization;Art;Machine learning;Text mining;Big data;Apache Spark;Classification;Naive Bayes},
  doi={10.1109/SIU.2017.7960721}}

}
```

```
@inproceedings{ulgen2017text,
  title     = {Text Classification with Spark Support Vector Machine},
  author    = {İskender Ülgen Oğul and Caner Ozcan and Özlem Hakdağlı},
  booktitle = {1st National Cloud Computing and Big Data Symposium (B3S’17)},
  year      = {2017},
  month     = {October},
  address   = {Antalya, Turkey},
  url       = {https://www.researchgate.net/publication/321579721_Text_Classification_with_Spark_Support_Vector_Machine}
}
```