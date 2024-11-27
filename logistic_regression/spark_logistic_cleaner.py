import pandas as pd
from pyspark.ml.feature import Tokenizer, HashingTF, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, udf, rand, col
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# --------------------------------------prepare data & tokenization-----------------------------------------------
spark = SparkSession.builder.appName('CC_cleaner')\
            .config('spark.executor.memory','16g')\
            .config('spark.driver.memory','16g')\
            .config('spark.local.dir', '/mnt/miaosen/tmp')\
            .config('spark.driver.maxResultsSize','0')\
            .getOrCreate()
train_df = spark.read.csv('/mnt/miaosen/crawl-text/cleaner_data/8G/train/train_passage.csv', inferSchema=True, header=True, escape='"')
# train_df = train_df.limit(int(train_df.count()/8000))  # about 500M
val_df = spark.read.csv('/mnt/miaosen/crawl-text/cleaner_data/8G/val/val_passage.csv', inferSchema=True, header=True, escape='"')
print(train_df.count(), val_df.count())
# visualize dataset statics
label_count_df = train_df.groupBy('label').count().toPandas()
label_count_df.plot(kind='bar')
plt.title('label Count')
plt.savefig('./final_train_dataset_statistic.png')
label_count_df = val_df.groupBy('label').count().toPandas()
label_count_df.plot(kind='bar')
plt.title('label Count')
plt.savefig('./final_val_dataset_statistic.png')
# pos and neg sample avg length
train_df = train_df.withColumn('length', length(train_df['data']))
train_df.groupBy('label').agg({'length':'mean'}).show()
val_df = val_df.withColumn('length', length(val_df['data']))
val_df.groupBy('label').agg({'length':'mean'}).show()

# tokenization dataset
tokenization = Tokenizer(inputCol='data', outputCol='tokenized_data')
train_df = tokenization.transform(train_df)
val_df = tokenization.transform(val_df)
tokenization.save('./final_model_passage/tokenizer')
# remove stop words
stop_words_removal = StopWordsRemover(inputCol='tokenized_data', outputCol='new_token')
train_df = stop_words_removal.transform(train_df)
val_df = stop_words_removal.transform(val_df)
stop_words_removal.save('./final_model_passage/stop_word_removal')
token_count = udf(lambda x: len(x), IntegerType())
train_df = train_df.withColumn('token_count', token_count(col('new_token')))
val_df = val_df.withColumn('token_count', token_count(col('new_token')))


# ---------------------------------------vectorize data----------------------------------------------------------
# Here we propose 3 way to vectorize data: A: count vector B:tf  C: tf-idf
# A: count vector
# count_vector = CountVectorizer(inputCol='new_token', outputCol='count_vector')
# train_count_vec = count_vector.fit(train_df).transform(train_df)
# val_count_vec = count_vector.fit(val_df).transform(val_df)
# count_vector.save('./count_vectorizer')
# B: tf vector
tf_vector = HashingTF(inputCol='new_token', outputCol='tf_vector')
train_tf_vec = tf_vector.transform(train_df)
val_tf_vec = tf_vector.transform(val_df)
tf_vector.save('./final_model_passage/tf_vectorizer')
# C: tf-idf vector
# tfidf_vector = IDF(inputCol='tf_vector', outputCol='tfidf_vector')
# IDF_model = tfidf_vector.fit(train_tf_vec)
# train_tfidf_vec = IDF_model.transform(train_tf_vec)
# val_tfidf_vec = IDF_model.transform(val_tf_vec)
# IDF_model.save('./final_model_passage/tfidf_vectorizer')
# assemble 3 way
# assembler = VectorAssembler(inputCols=['count_vector','token_count'], outputCol='X')
# train_count_vec = assembler.transform(train_count_vec)
# val_count_vec = assembler.transform(val_count_vec)
# assembler.save('./assembler_count')
assembler = VectorAssembler(inputCols=['tf_vector','token_count'], outputCol='X')
train_tf_vec = assembler.transform(train_tf_vec)
val_tf_vec = assembler.transform(val_tf_vec)
assembler.save('./final_model_passage/assembler_tf')
# assembler = VectorAssembler(inputCols=['tfidf_vector', 'token_count'], outputCol='X')
# train_tfidf_vec = assembler.transform(train_tfidf_vec)
# val_tfidf_vec = assembler.transform(val_tfidf_vec)
# assembler.save('./final_model_passage/assembler_tfidf')

#---------------------------------Logistic Regression---------------------------------------------------------
# train_1, test_1 = train_count_vec, val_count_vec
train_2, test_2 = train_tf_vec, val_tf_vec
# train_3, test_3 = train_tfidf_vec, val_tfidf_vec
# model_1 = LogisticRegression(featuresCol='X', labelCol='label').fit(train_1)
model_2 = LogisticRegression(featuresCol='X', labelCol='label').fit(train_2)
# model_3 = LogisticRegression(featuresCol='X', labelCol='label').fit(train_3)
# result_1 = model_1.evaluate(test_1).predictions
result_2 = model_2.evaluate(test_2).predictions
# result_3 = model_3.evaluate(test_3).predictions

# accuracy_1 = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(result_1)
accuracy_2 = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(result_2)
# accuracy_3 = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(result_3)

# precision_1 = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision').evaluate(result_1)
precision_2 = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision').evaluate(result_2)
# precision_3 = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision').evaluate(result_3)

# auc_1 = BinaryClassificationEvaluator(labelCol='label').evaluate(result_1)
auc_2 = BinaryClassificationEvaluator(labelCol='label').evaluate(result_2)
# auc_3 = BinaryClassificationEvaluator(labelCol='label').evaluate(result_3)

scores_df = pd.DataFrame({'feature_type':['Count_vec', 'TF_vec', 'TF-IDF_vec'],
                                             'accuracy':[0, accuracy_2, 0],
                                             'precision':[0, precision_2, 0],
                                             'auc':[0, auc_2, 0]})
print(scores_df)

# visualize
# plt.figure(figsize=(20, 6))
# plt.subplot(1, 3, 1)
# plt.title('Accuracy')
# plt.ylim(0.45, 0.95)
# scores_df.set_index('feature_type').accuracy.plot(kind='bar')
# plt.subplot(1, 3, 2)
# plt.title('Precision')
# plt.ylim(0.45, 0.95)
# scores_df.set_index('feature_type').precision.plot(kind='bar')
# plt.subplot(1, 3, 3)
# plt.title('AUC')
# plt.ylim(0.45, 1.0)
# scores_df.set_index('feature_type').auc.plot(kind='bar')
# plt.savefig('result.png')

# save model
# model_1.save('./count_vec')
model_2.save('./final_model_passage/tf_model')
# model_3.save('./final_model_passage/tf_idf_model')
