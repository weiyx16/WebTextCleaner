import argparse
from multiprocessing import Process
import os
import wget
import random
import gzip
import re
import numpy as np
from tqdm import tqdm
from langdetect import detect_langs
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from stopwords_bloom import stopwords as swb
from flagged_words import flagged_words as bdb
import json

def download_and_unzip(url, tmp_dir):
    tmp_f = os.path.join(tmp_dir, str(random.randint(1e8, 1e9-1)))
    downloaded = False
    for attempt in range(100):
        try:
            wget.download(url, tmp_f+'.gz')
            downloaded = True
        except:
            print("attempt error, retry" + str(attempt))
        else:
            break
    if downloaded:
        g_file = gzip.GzipFile(tmp_f+'.gz')
        open(tmp_f, 'wb+').write(g_file.read())
        g_file.close()
        os.system('rm -f '+tmp_f+'.gz')
        return tmp_f
    else:
        return 'Failed'

def get_len(text, lang):
    return len(text.split(" ")) if blank(lang) else len(text)

def get_first_word(text, lang):
    return text.split(" ")[0] if blank(lang) else text[0]

def get_word_list(text, lang):
    return text.split(" ") if blank(lang) else list(text)

def blank(lang):
    n = ['zh', 'zh-cn', 'zh-tw', 'ko', 'ja']
    return lang not in n

def clean_table(lines, lang):  # use hard rule to filter texts from tables
    to_delete = set()
    pointer_start = 0
    pointer = pointer_start+1
    while pointer_start<len(lines)-1:
        l_st = get_len(lines[pointer_start], lang)
        l_pt = get_len(lines[pointer], lang)
        while l_st==l_pt and pointer<len(lines)-1:
            pointer+=1
            l_pt = get_len(lines[pointer], lang)
        if pointer-3>pointer_start:
            for i in range(pointer_start, pointer):
                to_delete.add(i)
            pointer_start = pointer
            pointer+=1
        else:
            pointer_start+=1
            pointer= pointer_start+1
    
    pointer_start = 0
    pointer = pointer_start+1
    while pointer_start<len(lines)-1:
        l_st = get_first_word(lines[pointer_start], lang)
        l_pt = get_first_word(lines[pointer], lang)
        while l_st==l_pt and pointer<len(lines)-1:
            pointer+=1
            l_pt = get_first_word(lines[pointer], lang)
        if pointer-3>pointer_start:
            for i in range(pointer_start, pointer):
                to_delete.add(i)
            pointer_start = pointer
            pointer+=1
        else:
            pointer_start+=1
            pointer= pointer_start+1

    re_lines = []
    for i in range(len(lines)):
        if i not in to_delete:
            re_lines.append(lines[i])
    return re_lines

def end_mark(line, lang):
    if lang in ['ar', 'bn', 'el', 'fa', 'hi', 'ml', 'ne', 'pa', 'te', 'th', 'ur']:  # no end mark language
        return True
    if lang in ['zh', 'zh-cn', 'zh-tw', 'ja']:  # special mark language
        mark = ['。','，','？','”', '！', '：','；','……','；']
    else:  # latin 
        mark = ['.', '!', '...', '\"', '?', ':', ';', ',']
    for m in mark:
        if line.strip().endswith(m):
            return True
    return False
        
def jaccard_filter(lines, lang, rg = 3, fil=0.1):
    def jaccard_distance(sentA,sentB):
        s1 = set(get_word_list(sentA.lower(), lang))
        s2 = set(get_word_list(sentB.lower(), lang))
        return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    to_delete = set()
    for i, line in enumerate(lines):
        for j in range(1,rg+1):
            if i>=j and jaccard_distance(line, lines[i-j])>fil:
                to_delete.add(i)
                to_delete.add(i-j)
    
    re_lines = []
    for i in range(len(lines)):
        if i not in to_delete:
            re_lines.append(lines[i])
    return re_lines

class match_filter():
    def __init__(self):
        self.language_dict = {'ar': 'Arabic', 'bn': 'Bengali', 'da': 'Danish', 'fi': 'Finnish', 'el': 'Greek',
                    'hu': 'Hungarian', 'kk': 'Kazakh', 'pt': 'Portuguese', 'ru': 'Russian', 'sv': 'Swedish',
                    'az': 'Azerbaijani', 'ca': 'Catalan', 'nl': 'Dutch', 'fr': 'French', 'he': 'Hebrew',
                    'id': 'Indonesian', 'ne': 'Nepali', 'sl': 'Slovene', 'tg': 'Tajik', 'eu': 'Basque',
                    'zh-cn': 'Chinese', 'en': 'English', 'de': 'German', 'hi': 'Hinglish', 'it': 'Italian',
                    'no': 'Norwegian', 'ro': 'Romanian', 'es': 'Spanish', 'tr': 'Turkish'}
        self.bad_word = json.load(open("bad_words.json", 'r'))
        for k, v in bdb.items():
            if k in self.bad_word.keys():
                self.bad_word[k].extend(v)
            else:
                self.bad_word[k] = v

    def stop_word_removal(self, texts, lang):
        if not lang in self.language_dict.keys():
            return False   # not suport language, skip.
        stop_words = list(stopwords.words(self.language_dict[lang].lower()))
        if lang in swb.keys():
            stop_words.extend(swb[lang])
        stop_words = set(stop_words)
        stop_word_count = 0
        for w in stop_words:
            stop_word_count += texts.count(w)
        words = get_len(texts, lang)
        stop_word_ratio = stop_word_count / words
        return stop_word_ratio<0.3 if lang=='en' else stop_word_ratio<0.15

    def flagged_word_removal(self, texts, lang):
        if not lang in self.bad_word.keys():
            return False # not support for that language
        badwords = set(self.bad_word[lang])
        for w in badwords:
            if blank(lang) and w in get_word_list(texts, lang):
                return True
            if not blank(lang) and w in texts:
                return True
        return False

mf = match_filter()
def document_filter(doc):
    skip = ['{', '<',]
    if len(doc)<5:
        return None, False
    docs = doc[:100] if len(doc)>100 else doc
    texts = '\n'.join(docs)
    for w in skip:
        if w in texts:
            return None, False
    try:
        lang = detect_langs(texts)[0]
    except:
        return None, False
    if lang.prob < 0.99:
        return None, False
    lang = lang.lang
    if mf.stop_word_removal(texts, lang[:2]):
        return None, False
    if mf.flagged_word_removal(texts, lang[:2]):
        return None, False
    return lang, True
    
def rule_clean(content, lang):
    bad_sentence = ['http:','https:', '.com', '@', 'javascrip','|', '©', '&', '﻿', '', '｜']
    content_clean = []
    for line in content:
        if get_len(line, lang) < 5:
            continue
        if lang=='en' and not bool(re.search('[a-z]', line)):
            continue
        if not end_mark(line, lang):
            continue
        flag = False
        for w in bad_sentence:
            if w in line:
                flag = True
                break
        if flag:
            continue
        content_clean.append(line)
    return content_clean

def first_clean(tmp_f, tgt_f, is_main=False):
    f = open(tmp_f, 'r').read()
    blocks = f.split('WARC/1.0')
    tgt_blocks = dict()
    for b in tqdm(blocks):
        info_content = b.strip().split('\n\n')
        if not len(info_content)==2:
            continue
        info = info_content[0]
        contents = info_content[1].split('\n')
        if 'Content-Type' in info and not 'Content-Type: text/plain' in info:
            continue
        # if 'WARC-Identified-Content-Language' in info and not 'WARC-Identified-Content-Language: eng\n' in info :
        #     continue
        lang, st = document_filter(contents)
        if not st:   # filter and remove whole document 
            continue
        content_clean = rule_clean(contents, lang)
        content_clean = clean_table(content_clean, lang)
        content_clean = jaccard_filter(content_clean, lang)
        if lang not in tgt_blocks.keys():
            tgt_blocks[lang] = [content_clean]
        else:
            tgt_blocks[lang].append(content_clean)
    # if is_main and len(tgt_blocks)>0:
    #     print("examples:", tgt_blocks[0])

    for lang, tblocks in tgt_blocks.items():
        fs = tgt_f.split("/")
        fs.insert(-2,lang)
        langp = '/'.join(fs)
        pre_path = '/'.join(fs[:-1])
        if not os.path.exists(pre_path):
            os.makedirs(pre_path)
        with open(langp, 'w') as f:
            for blocks in tblocks:
                if len(blocks)>3:
                    f.writelines([l + '\n' for l in blocks])
                    f.writelines(['\n', '\n'])
    assert False
    return 

def get_args_parser():
    parser = argparse.ArgumentParser('get information', add_help=False)
    parser.add_argument('--path-id', default='', type=str)
    parser.add_argument('--process-num', default=1, type=int)
    parser.add_argument('--tmp-dir', default='./tmp', type = str)
    parser.add_argument('--output-dir', default='./output', type = str)
    return parser

def main(files_to_download, tmp_dir, target_path, is_main):
    """
    files_to_download: files to download, list of https url that can directly download data
    tmp_dir: temp dir for download and unzip
    target_path: target path to save the files, a target path list corresponding to files_to_download
    return
    None
    """
    iterator = tqdm(enumerate(files_to_download)) if is_main else enumerate(files_to_download)
    for i, f in iterator:
        tmp_file_name = download_and_unzip(f, tmp_dir)
        if tmp_file_name=='Failed':
            print("error happened in downloading file", f, " we will skipping it.")
            continue
        first_clean(tmp_file_name, target_path[i], is_main)
        # os.system('rm -f '+tmp_file_name)
    return

def get_args(args, i):
    """
    split data to multiprocess process args
    """
    out_args = dict()
    out_args['tmp_dir'] = args.tmp_dir
    files = open(os.path.join('path',args.path_id), 'r').readlines()
    files_per_process = len(files)//args.process_num+1
    if i==0:
        print("per process files:", files_per_process)
    process_files = files[i*files_per_process: (i+1)*files_per_process]
    out_args['files_to_download'] = ['https://data.commoncrawl.org/'+f.replace('\n', '') for f in process_files]
    f_id = [f.split('/')[-1].split('.')[0] for f in process_files]
    # if not os.path.exists(os.path.join(args.output_dir, args.path_id)):
    #     os.mkdir(os.path.join(args.output_dir, args.path_id))
    out_args['target_path'] = [os.path.join(args.output_dir, args.path_id, f+'.txt') for f in f_id]
    return (out_args['files_to_download'],out_args['tmp_dir'],out_args['target_path'], i==0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('get information', parents=[get_args_parser()])
    args = parser.parse_args()
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for i in range(args.process_num):
        Process(target=main, args=get_args(args, i)).start()
    
