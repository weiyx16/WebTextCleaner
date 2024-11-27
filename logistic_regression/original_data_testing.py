import pandas as pd
from pyspark.ml.feature import Tokenizer, HashingTF, StopWordsRemover, CountVectorizer, IDFModel, VectorAssembler
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import length, udf, rand, col, element_at, split
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os
from tqdm import tqdm
import numpy as np
import seaborn as sns
from time import time
from random import sample
class LR:
    def __init__(self, tok_dir, stop_wr_dir, tfvec_dir, idfvec_dir, assem_dir ,md_dir,):
        self.tokenizer = Tokenizer.load(tok_dir)
        self.stop_word_removal = StopWordsRemover.load(stop_wr_dir)
        self.model = LogisticRegressionModel.load(md_dir)
        self.TF_vectorizer = HashingTF.load(tfvec_dir)
        # self.IDF_vectorizer = IDFModel.load(idfvec_dir)
        self.assembler = VectorAssembler.load(assem_dir)
        self.token_count = udf(lambda x: len(x), IntegerType())
        # self.evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
    
    def predict_batch(self, df):
        """
        df: dataframe with col: data[pain text] label[0(dirty) or 1(clean)]
        """
        batch_size = df.count()
        df = self.tokenizer.transform(df)
        df = self.stop_word_removal.transform(df)
        df = df.withColumn('token_count', self.token_count(col('new_token')))
        df = self.TF_vectorizer.transform(df)
        # df = self.IDF_vectorizer.transform(df)
        df = self.assembler.transform(df)
        result = self.model.evaluate(df).predictions
        # accuracy = self.evaluator.evaluate(result)
        res = re_organize_res(result)
        return batch_size, res

def re_organize_res(res):
    # res pandas dataframe
    # res['score'] = res['probability'].apply(lambda x: x[1])
    # PNcount = res.groupBy('prediction').count()
    S_list = res.select('probability').rdd.flatMap(lambda x: x).collect()
    p_list = res.select('prediction').rdd.flatMap(lambda x: x).collect()
    Slist = [float(x[1]) for x in S_list]
    return Slist, sum(p_list)

def lines2passage(ls, spark, lab):
    blocks = []
    tmp = ''
    for l in ls:
        if l == '\n':
            if tmp.strip()=='':
                tmp = ''
                continue
            blocks.append(Row(data=tmp, label=lab))
            tmp = ''
        else:
            tmp=tmp+l.strip('\n')
    try:
        pd = spark.createDataFrame(blocks)
        stat = True
    except:
        pd = None
        stat = False
    return pd, stat

def main():
    MAX_FILE = 128
    folder_path = '/mnt/miaosen/crawl-text/stories'
    ignore_file = ['stories.1.txt','stories.2.txt']
    flist = os.listdir(folder_path)
    for ff in ignore_file:
        flist.remove(ff)
    flist = flist if len(flist)<=MAX_FILE else flist[:128]
    # folder_path = '/mnt/miaosen/nlp_data/common_crawl'
    # year = '2015'
    # fdlist = []
    # for fd in os.listdir(folder_path):
    #     if year in fd:
    #         fdlist.append(fd)
    # flist = []
    # for fd in fdlist:
    #     subpath = os.path.join(folder_path, fd)
    #     txts = sample(os.listdir(subpath), MAX_FILE)
    #     flist.extend([os.path.join(subpath, f) for f in txts])
    datasetname = 'stories_final'
    label = 1
    
    spark = SparkSession.builder.appName('CC_cleaner')\
            .config('spark.executor.memory','16g')\
            .config('spark.driver.memory','16g')\
            .config('spark.local.dir', '/mnt/miaosen/tmp')\
            .config('spark.driver.maxResultsSize','0')\
            .getOrCreate()
    

    logisticmodel = LR('./final_model_passage/tokenizer', './final_model_passage/stop_word_removal', 
                        './final_model_passage/tf_vectorizer', './final_model_passage/tfidf_vectorizer',
                        './final_model_passage/assembler_tf', './final_model_passage/tf_model')

    Total_Batch_Size = 0
    POS_sample = 0
    Scores_List = []
    st = time()
    for f in tqdm(flist):
        path = os.path.join(folder_path, f)
        fio = open(path, 'r')
        lines = fio.readlines() 
        passages, stat = lines2passage(lines, spark, label)
        if not stat:
            continue
        bs, (S_list, POS) = logisticmodel.predict_batch(passages)
        POS_sample += POS
        Total_Batch_Size += bs
        Scores_List.extend(S_list)
    print('testing time:', time()-st)
    print(f'POS samples:{POS_sample}, NEG samples: {Total_Batch_Size-POS_sample}')
    print(f'Batchsize: {Total_Batch_Size}')
    print('mean of scores:', np.mean(Scores_List))
    print('std of scores:', np.std(Scores_List))
    if len(Scores_List)>10000:
        ls = list(sample(Scores_List, 10000))
        sns.histplot(ls, kde=True)
    else:
        sns.histplot(Scores_List, kde=True)
    plt.savefig(f'./data_info/{datasetname}.png')
    np.save(f'./data_info/{datasetname}.npy', np.array(Scores_List))

if __name__=='__main__':
    main()