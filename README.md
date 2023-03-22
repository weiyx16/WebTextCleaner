# Web Dataset Script

## Key Usage:
+ 0318 version Rules based clean:  
    `python main.py --path-id CC-MAIN-2020-50 --process-num 60 --output-dir /output/common_crawl_cleaned/202050_v20230318_subset3of4 --subset 3/4`
+ 0321 version With rules better reviewed: ruleset-1 an ruleset-2

## Pipeline

+ 0317  
    + Pick a select subset: 2020.50 CC (7.9TB source file)  
    + Writing Rules
        + Rule-1 [done] 
        + Rule-2 [done][double_check, is that too general? like sex/shit]
        + Rule-3 [done]
        + Rule-4 [done]
        + Rule-5 [==Rule-2]
        + Rule-6 [done]
        + Rule-7 [done]
        + Rule-8 [done][double_check]
        + Rule-9 [done]
        + Rule-10 [TODO][double_check][how: By tokenizer?]
        + Rule-11 [done]
        + Rule-12 [==Rule-24]
        + Rule-13 [done]
        + Rule-14 [done]
        + Rule-15 [done]
        + Rule-16 [done]
        + Rule-17 [done]
        + Rule-18 [done]
        + Rule-19 [done]
        + Rule-20 [done]
        + Rule-21 [done]
        + Rule-22 [done]
        + Rule-23 [pass]
        + Rule-24 [done]
        + Rule-25 [done]
    + Check the lower/upper problem [done]

+ 0319-0321  
    + Summary and compare ruleset-1/2/3/4 and double check code (e.g. add langdetect for each line) [done]
    + Check results from first 5k subsubset [done]
    + Subset support [done]
    + **Submit** CC202050 Rule0318 Subset\[0/4\] [done]
        + 1/4 subset; 8C64(Eadsv5) ~ 7h
        + should : 18000 subsets; final: 6970 subsets; total: ~140G
        + Rerun for the left: 18000 subsets; final: 13060 subsets; total: ~260G; the error is `HTTP Error 503: Service Unavailable`
        + ReRerun for the left: 18000 subsets; final: 16783 subsets; total: ~336G; 
    + Double check the current performances with 1.4B + 5GB data like Gopher [todo]
    + Split OSCAR2201 and Other202050 [done]
        + Run GPT on theirs [done]
    + Split ours [done]
        + Run GPT on ours 0of4 [done]
        + Why the validation loss is so high?
            + Subset Effect: run on 2of4: not the reason [done]
            + Do some unit test first. [done]
            + Simply the rule set with different parts and validate it one-by-one. 
                + Improved Ruleset-1 : normal [done]
                + Improved Ruleset-2 : normal [done]

    + Repeat Removal from Gopher [todo][skip it]
    + Linear Classifier [todo][skip it]


+ 0322 
    + Deduplication: among train [todo]  
    + Deduplication: train v.s. test [todo]  

+ Others  
    + need random noise?  

+ Other Sources
    + Github  
    + Wikipedia  
    + Pile-of-Law  

> Compute
```bash
TARGET_NAME    RESOURCE_GROUP          SUBSCRIPTION             SERIES
-------------  ----------------------  -----------------------  -------------------------------------------
msroctovc      gcr-singularity-octo    Singularity Shared OCTO  NDv2g1
msrresrchvc    gcr-singularity-resrch  Singularity Shared       Eadsv5, Ev3, NCv2, NCv3, ND, NDAMv4, NDv2g1

SERIES    ASSOCIATED_TYPES
--------  ----------------------------------------------------
Eadsv5    8C4, 8C8, 8C16, 8C20, 8C32, 8C48, 8C64
Ev3       8C1, 8C2, 8C4, 8C8, 8C16, 8C32
NCv2      16G1-P100, 16G2-P100, 16G4-P100, 16G4-P100-IB
NCv3      16G1-V100, 16G2-V100, 16G4-V100, 16G4-V100-IB
ND        24G1-P40, 24G2-P40, 24G4-P40, 24G4-P40-IB
NDAMv4    80G1-A100, 80G2-A100, 80G4-A100, 80G8-A100-IB-NvLink
NDv2g1    16G1-V100, 16G2-V100, 16G4-V100, 16G8-V100
```

## Original Readme

[C4](https://www.tensorflow.org/datasets/catalog/c4) is a great way to get a colossal cleaned web corpus. Unfortunately, Google open-sourced c4 script highly depends on GCP and code mixed in a big repo. Therefore, it takes work to develop it freely. This repository extracts the processing logic and implements it to run on Spark. In addition, some helpful data process method in MassiveText is implemented in massivetext_utils.py.

## Run c4 script on Spark

Setup c4 work environment.

```bash
# 1. Create an independent Anaconda environment and install python dependencies
conda create -y -n c4-env conda-pack && conda activate c4-env
pip install git+https://github.com/shjwudp/c4-dataset-script

# 2. Download punkt tokenizer
python -m nltk.downloader -d $(which python | xargs dirname)/../nltk_data punkt

# 3. Run pyspark requires JAVA to be installed in your environment, you should
#    make sure you have JDK installed and JAVA_HOME configured.
```

If everything goes well, you can make the C4 dataset on localhost.

```bash
python -m c4_dataset_script.c4_script --wet-file-paths $PATH_TO_YOUR_CC_WET_FILE
```

Or submit to spark cluster.

```bash
# 1. Before submitting to the cluster, you need to package the environment conda env
conda pack --name c4-env -o c4-env.tar.gz

# 2. Submit to spark cluster
PYSPARK_DRIVER_PYTHON=python \
PYSPARK_PYTHON=./environment/bin/python \
python c4_dataset_script/c4_script.py \
    --wet-file-paths $PATH_TO_YOUR_CC_WET_FILE \
    --c4-save-path $PATH_TO_YOUR_C4_OUTPUT \
    --spark-master $SPARK_MASTER_ADDR \
    --spark-archives c4-env.tar.gz#environment
```

## Make colossal cleaned Chinese web corpus

Referring to the method of C4, there is a data processing pipeline building for a cleaned Chinese web corpus. It includes web page download, Chinese recognition, heuristics text filter method, toxic recognition and filter, and Repetition Removal used in Google/DeepMind MassiveText.

## 1. Download the WET crawl archive index file

Common Crawl organized crawled data into some archives. You can browse the archives list from [here](https://commoncrawl.org/the-data/get-started/). In the next step, we will download text data (WET) as the input of processing. First, download the WET crawl archive index file.

```bash
cd c4_dataset_script
wget -r --no-parent https://data.commoncrawl.org/crawl-data/${CRAWL_ARCHIVE_ID}/wet.paths.gz
```

*You can get CRAWL_ARCHIVE_ID [here](https://commoncrawl.org/the-data/get-started/). For instance: CC-MAIN-2022-49.*

## 2. Run download and Chinese screening script on Spark

```bash
spark-submit --master ${SPARK_MASTER_ADDR} \
    Chinese/download_web_docs.py \
        --wet-paths ./data.commoncrawl.org/crawl-data/${CRAWL_ARCHIVE_ID}/wet.paths.gz \
        --output ./download-docs
```

## 3. Filter out non-sentence lines and toxic document

Refer to the c4 heuristics method. I used the following strategies for cleaning up Common Crawl's web-extracted text:

 - Only retained lines that ended in a terminal punctuation mark or colon.
 - Discarded any page with fewer than five sentences and only retained lines that
contained at least five words.
 - Removed any page that contained any word on the "List of Dirty, Naughty, Obscene
or Otherwise Bad Words."
 - Many of the scraped pages contained Chinese garbled, so we removed any line with the garbled characters. For example: "[-]|□|■|�".

```bash
cat ./download-docs/*/part-* | \
    python Chinese/filter_out_bad_lines.py \
        --badwords_filepath ./badwords/zh \
         > clean_docs.jsonl
```

*About 93.57% of documents are filtered out in this stage. You can see samples of filtered documents [here](data/Chinese_bad-lines_samples.jsonl).*

## 4. Repetition Removal

Check the percentage of duplicate content in the web document, and the program will remove documents whose duplicate proportion exceeds the preset threshold. This function implements "Repetion Removal" as described in [Gopher](https://arxiv.org/abs/2112.11446).

```bash
spark-submit --master ${SPARK_MASTER_ADDR} \
    Chinese/repetition_removal.py \
        --input clean_docs.jsonl \
        --output ./repetition_removal_output
```

*About 21.21% of documents are filtered out in this stage. You can see samples of filtered documents [here](data/Chinese_Repetition-Removal_samples.jsonl).*
