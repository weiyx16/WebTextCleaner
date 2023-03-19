"""
    Usage: python main.py --path-id CC-MAIN-2020-50 --process-num 8 --tmp-dir ./tmp --output-dir ./output
    Target: Download and process the Common Crawl dataset with rule-based filtering
    Pipeline:
        1. Download the dataset
        2. Filter the dataset with rule-based filtering
        3. Clean the cached files
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import os
import re
import wget
import json
import gzip
import html
import ftfy
import random
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

# languag detect
from langdetect import detect_langs
import fasttext
if not os.path.exists('lib.176.bin'):
    wget.download('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', 'lib.176.bin')
    wget.download('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz', 'lib.176.ftz')
LanDetectModel = fasttext.load_model('lib.176.bin')

# stop words
import spacy
en = spacy.load('en_core_web_sm')
swb_spacy = en.Defaults.stop_words
from nltk.corpus import stopwords as swb_nltk
from RuleBased.stopwords_bloom import stopwords as swb_bloom

# bad words
from RuleBased.flagged_words import flagged_words as bdb

# ascii
import string
# for other non-ascii label: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c
printable = string.printable
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# stop words
swb_custom = ['(', ')', '\'', '"', '[', ']', ',', '.', '“', '``', "''",
                     '”', ':', ';', '?', '!', '<', '>', 's', '-s', '-ly', '<s>', '</s>']
STOP_WORDS = swb_spacy | set(
    swb_nltk.words('english')) | set(swb_bloom['en'])

STOP_WORDS_FREQ = set(['the', 'be', 'to', 'of', 'and', 'that', 'have', 'with'])

# --------------
# HYPERPARAMETERS
STOP_WORD_RATIO = 0.3 # if the ratio of stop words is less than this number, we will filter it
STOP_WORD_COUNT = 2   # if the number of stop words is less than this number, we will filter it. [notice we have different stop word definition]
MIN_SENTENCE_INPAGE = 5 # if the sentence number is less than this number, we will filter it
MIN_SENTENCE_LENGTH = 3 # if the token number in the sentence is too small, then we filter it.
DOCUMENT_LENGTH_RANGE = [50, 100000] # document max/min token numbers range
WORD_LENGTH_RANGE = [3, 10]          # word max/min character numbers range
BWBullet = 0.9          # if the sentence portition starts with bullet > BWBullet, we filter it
EWEllipsis = 0.3        # if the sentence portition ends with ellipsis > EWEllipsis, we filter it
SymbolRatio = 0.1       # hash tag or ellipsis ratio in the document
JAC_MIN = 0.5           # jaccard_distance distance to measure the similarity between consecutive lines
WORD_WITH_ALPHABETIC = 0.8 # the min ratio that for each word with at least one alphabetic
CODE_WEBPAGE_FLAG = [r'{', r'<', r'lorem ipsum'] ## TODO: DOUBLE CHECK: we may want to keep the pages like stack overflow which may contains code
MAX_CODE_WEBPAGE_COUNT = 2 # code webpage flag appears > MAX_CODE_WEBPAGE_COUNT, the document will be filtered
RANDOM_CASES = 0.0 # leave some noise code TODO: check will should we add it
isZH = set(['zh', 'zh-cn', 'zh-tw', 'ko', 'ja'])    
bad_tokens = ['http:','https:', '.com', '@', 'javascript','|', '©', '&', '﻿', '', '｜']
# --------------


def get_len(text, lang):
    return len(text.split(" ")) if blank(lang) else len(text)

def get_character_len(text, lang):
    return sum([len(ele) for ele in text.split(" ")]) if blank(lang) else 4

def get_first_word(text, lang):
    return text.split(" ")[0] if blank(lang) else text[0]

def get_word_list(text, lang):
    return text.split(" ") if blank(lang) else list(text)

def blank(lang):
    return lang not in isZH


class match_filter():
    def __init__(self):
        self.language_dict = {'ar': 'Arabic', 'bn': 'Bengali', 'da': 'Danish', 'fi': 'Finnish', 'el': 'Greek',
                    'hu': 'Hungarian', 'kk': 'Kazakh', 'pt': 'Portuguese', 'ru': 'Russian', 'sv': 'Swedish',
                    'az': 'Azerbaijani', 'ca': 'Catalan', 'nl': 'Dutch', 'fr': 'French', 'he': 'Hebrew',
                    'id': 'Indonesian', 'ne': 'Nepali', 'sl': 'Slovene', 'tg': 'Tajik', 'eu': 'Basque',
                    'zh-cn': 'Chinese', 'en': 'English', 'de': 'German', 'hi': 'Hinglish', 'it': 'Italian',
                    'no': 'Norwegian', 'ro': 'Romanian', 'es': 'Spanish', 'tr': 'Turkish'}
        self.bad_word = json.load(open("RuleBased/bad_words.json", 'r'))
        for k, v in bdb.items():
            if k in self.bad_word.keys():
                self.bad_word[k].extend(v)
            else:
                self.bad_word[k] = v
        self.badwords_en = set(self.bad_word['en'])

    def stop_word_removal(self, texts, lang):
        """
            Notice that: we assume input is en now
            `Rule-22`: Frequent stop word minimal
            `Rule-24`: Stop word maximum

            @params:   
                Input:   
                    - texts: a single line (string)   
                    - lang: language of the text   
                Output:   
                    - bool: True: pass, False: fail   
        """
        if not lang in self.language_dict.keys():
            return False   # not suport language, skip.
        # stop_words = list(stopwords.words(self.language_dict[lang].lower()))
        # if lang in swb.keys():
        #     stop_words.extend(swb[lang])
        # stop_words = set(stop_words)
        stop_word_count = 0
        for w in STOP_WORDS:
            stop_word_count += texts.count(w)
        stop_word_freq_count = 0
        for w in STOP_WORDS_FREQ:
            stop_word_freq_count += texts.count(w)
        total_words = get_len(texts, lang)
        if total_words == 0:
            return False
        stop_words_ratio = stop_word_count / total_words

        # double check if fail or success
        SucceedinSWR = stop_words_ratio > STOP_WORD_RATIO # if lang == 'en' else stop_word_ratio < 0.15
        SucceedinSWC = stop_word_freq_count >= STOP_WORD_COUNT

        return SucceedinSWC and SucceedinSWR

    def flagged_word_removal(self, texts, lang):
        """
            Notice that: we assume input is en now
            `Rule-2`: Bad word detector removal

            @params:
                Input:
                    - texts: a single line (string)
                    - lang: language of the text
                Output:
                    - bool: True: pass, False: fail
        """
        if not lang in self.bad_word.keys():
            return False # not support for that language
        # badwords = set(self.bad_word[lang])
        list_of_texts = set(get_word_list(texts, lang))
        for w in self.badwords_en:
            if blank(lang) and w in list_of_texts:
                return False
            if not blank(lang) and w in texts:
                return False
        return True
    
MF = match_filter()


def get_args_parser():
    parser = argparse.ArgumentParser('get information', add_help=False)
    parser.add_argument('--path-id', default='', type=str)
    parser.add_argument('--subset', default='0/1', type = str, help='x/y means the xth subset in y subsets')
    parser.add_argument('--process-num', default=1, type=int)
    parser.add_argument('--tmp-dir', default='./tmp', type = str)
    parser.add_argument('--output-dir', default='./output', type = str)
    return parser


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


def language_detect(sentence):
    """
        A comparison on language detect: https://modelpredict.com/language-identification-survey#reported-metrics
        fastText is faster and more accurate
        Rule-1
    """
    lang = LanDetectModel.predict(sentence)
    is_en = False
    try:
        if lang[0][0].endswith('en'):
            # prob: lang[1][0]
            is_en = True
    except:
        pass
    
    # detect with detect_lans: maybe faster
    # try:
    #     lang = detect_langs(sentence)[0]
    # except:
    #     is_en = False
    
    # if lang.prob < 0.99:
    #     # we only keep the language that
    #     is_en = False
    # lang = lang.lang
    return is_en


def clean_table(lines, lang):  
    """
        Rules 16: remove the tables
    """
    # use hard rule to filter texts from tables
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
    """
        Rule-3: end with terminate flags
    """
    if lang in ['ar', 'bn', 'el', 'fa', 'hi', 'ml', 'ne', 'pa', 'te', 'th', 'ur']:  # no end mark language
        return True
    if lang in ['zh', 'zh-cn', 'zh-tw', 'ja']:  # special mark language
        mark = ['。','，','？','”', '！', '：','；','……','；']
    else:  # latin 
        mark = ['.', '!', '...', '\"', '?', ':', ';', ',', '-', "\'"]
    for m in mark:
        if line.strip().endswith(m):
            return True
    return False


def jaccard_distance(sentA, sentB, lang):
    s1 = set(get_word_list(sentA.lower(), lang))
    s2 = set(get_word_list(sentB.lower(), lang))
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2))) 


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def text_clean(text):
    # Rule-13: clean blank, clean HTML tags; but emoji?
    # special unicode
    unicode_map = {'\u2013': '-', '\u2014': '-', '\u00ae': ' ', '\u2026': '...', '&#034;': '"', '&#039;': '\'', '&quot;': '"', '&amp;amp;': '&', '&amp;': '&', '&hellip;': '…', '&ndash;': '-', '&#10;': ' ', '&#13;': ' ', '&#10073': ' ', '&nbsp;': ' ',
                    '<em>': ' ', '</em>': ' ', '<br>': ' ', '</br>': ' ', '<i>': ' ', '</i>': ' ', '<p>': ' ', '</p>': ' ', '<span>': ' ', '</span>': ' ',
                    '<EM>': ' ', '</EM>': ' ', '<BR>': ' ', '</BR>': ' ', '<I>': ' ', '</I>': ' ', '<P>': ' ', '</P>': ' ', '<SPAN>': ' ', '</SPAN>': ' ',
                    '\xa0': ' ', '\t': ' ', '\u200b': ' ', '\u2005': ' ', '\u2009': ' ', '\u2028': ' ', '\u00e8': ' '}
    for k, v in unicode_map.items():
        text = text.replace(k, v)

    TAG_RE = re.compile(r'<[^>]+>')
    text = TAG_RE.sub('', text)
    text = basic_clean(text)
    text = whitespace_clean(text)
    return text


def document_filter(doc):
    """
        @params:
            Input:
                - doc: list of strings in the block
    """
    # Rule-4: skip if the web page is too short
    if len(doc) < MIN_SENTENCE_INPAGE:
        return None, False

    # Rule-7&8: remove the web page if it contains "{" or "<"; TODO: double check
    texts = '\n'.join(doc)
    for w in CODE_WEBPAGE_FLAG:
        if texts.lower().count(w) > MAX_CODE_WEBPAGE_COUNT:
            return None, False
        
    # Rule-1: language detection: notice that we only reserve en samples
    is_en = language_detect(' '.join(doc))
    if is_en:
        lang = 'en'
    else:
        return None, False
    
    # Rule-4: skip if the sentence is too short
    filtered_doc = ['', '']  # init with two empty string for rule-9
    tokensWithAlphabetic = 0
    total_tokens = 0
    total_characters = 0
    for Ele in doc:
        ele = Ele.lower()
        sent_length = get_len(ele, lang)
        # Rule-4: skip if the sentence is too short
        if sent_length >= MIN_SENTENCE_LENGTH:
            # Rule-6: filter out javascript in sentence
            if language_detect(Ele) and 'javascript' not in ele:
                # Rule-11: web code 
                if not any(w in ele for w in bad_tokens):
                    # Rule-3: end with termination.
                    if end_mark(ele, lang):
                        # Rule-21: alphabet in words
                        # and not bool(re.search('[a-z]', ele))
                        _ele = [''.join(filter(lambda x: x in alphabet, _token)) for _token in get_word_list(ele, lang)]
                        for _token in _ele:
                            if len(_token) > 0:
                                tokensWithAlphabetic += 1

                        # Rule-9&15: when three lines are the same, pop one
                        if (ele == filtered_doc[-1] and ele == filtered_doc[-2]) or (jaccard_distance(ele, filtered_doc[-1], lang) > JAC_MIN and jaccard_distance(ele, filtered_doc[-2], lang) > JAC_MIN):
                            # pop one line
                            popline = filtered_doc[-1]
                            filtered_doc = filtered_doc[:-1]
                            total_tokens -= get_len(popline, lang)
                            total_characters -= get_character_len(popline, lang)
                        else:
                            # Rule-14: uppercase the first characteristic
                            filtered_doc.append(Ele.strip().capitalize())
                            total_tokens += sent_length
                            total_characters += get_character_len(ele, lang)
    filtered_doc = filtered_doc[2:] # pop the first two placehold

    # Rule-17: document token words
    if total_tokens < DOCUMENT_LENGTH_RANGE[0] or total_tokens > DOCUMENT_LENGTH_RANGE[1]:
        return None, False
    
    # Rule-18: mean token lengths
    if total_characters < WORD_LENGTH_RANGE[0] * total_tokens or total_characters > WORD_LENGTH_RANGE[1] * total_tokens:
        return None, False

    # Rule-21: words with alphabet
    if tokensWithAlphabetic < total_tokens * WORD_WITH_ALPHABETIC:
        return None, False
    
    # Rule-14: remove the tables
    filtered_doc = clean_table(filtered_doc, lang)

    # Rule-20: not too much begin with bullet and end with ellipsis
    # Rule-19: symbol-to-word ratio
    beginWithBullet = 0
    endWithEllipsis = 0
    EllipsisCount = 0
    HashCount = 0
    filtered_cleaned_doc = []
    for ele in filtered_doc:
        if ele.endswith('...'):
            endWithEllipsis += 1
        if ele.startswith('*') or ele.startswith('+') or ele.startswith('-') or ele.startswith('#') or ele.startswith('>'):
            beginWithBullet += 1
        EllipsisCount += ele.count('...')
        HashCount += ele.count('#')
        # Rule-13/14: line cleaning
        ele = text_clean(ele)
        filtered_cleaned_doc.append(ele)
    if beginWithBullet > BWBullet * len(filtered_doc) or endWithEllipsis > EWEllipsis * len(filtered_doc):
        return None, False
    if EllipsisCount > SymbolRatio * total_tokens or HashCount > SymbolRatio * total_tokens:
        return None, False

    texts = '\n'.join(filtered_cleaned_doc)
    if not MF.stop_word_removal(texts, lang[:2]):
        # rule-22&24
        return None, False
    if not MF.flagged_word_removal(texts, lang[:2]):
        # rule-2
        return None, False
    return lang, texts


def detect_and_clean(tmp_f, tgt_f):
    """
        Header of WARC file:
        WARC/1.0
        WARC-Type: warcinfo
        WARC-Date: 2020-12-06T05:37:50Z
        WARC-Filename: CC-MAIN-20201123153826-20201123183826-00000.warc.wet.gz
        WARC-Record-ID: <urn:uuid:29049de7-c6ab-4e46-b0e2-428168da10d3>
        Content-Type: application/warc-fields
        Content-Length: 382
    """
    f = open(tmp_f, 'r').read()
    blocks = f.split('WARC/1.0')

    tgt_blocks = dict()
    for b in tqdm(blocks):
        info_content = b.strip().split('\n\n')
        if not len(info_content) == 2:
            # a standard block for content should have 2 parts
            # info head \n\n content
            continue

        info = info_content[0]
        contents = info_content[1].split('\n')
        if 'Content-Type' in info and not 'Content-Type: text/plain' in info:
            continue

        # if 'WARC-Identified-Content-Language' in info and not 'WARC-Identified-Content-Language: eng\n' in info :
        #     continue
        
        lang, st = document_filter(contents)
        if not isinstance(st, str):   # filter and remove whole document 
            continue
        content_clean = st

        if lang not in tgt_blocks.keys():
            tgt_blocks[lang] = [content_clean]
        else:
            tgt_blocks[lang].append(content_clean)
    # if is_main and len(tgt_blocks)>0:
    #     print("examples:", tgt_blocks[0])

    for lang, tblocks in tgt_blocks.items():
        fs = tgt_f.split("/")
        fs.insert(-2, lang)
        langp = '/'.join(fs)
        pre_path = '/'.join(fs[:-1])
        if not os.path.exists(pre_path):
            os.makedirs(pre_path)
        with open(langp, 'w') as f:
            for blocks in tblocks:
                # if len(blocks) > 3:
                #     f.writelines([l + '\n' for l in blocks])
                #     f.writelines(['\n', '\n'])
                f.writelines(blocks)
                f.writelines(['\n', '\n'])
    return 


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
        if tmp_file_name == 'Failed':
            print("error happened in downloading file", f, " we will skipping it.")
            continue
        detect_and_clean(tmp_file_name, target_path[i])
        # we need to remove the tmp file
        os.system('rm -f '+tmp_file_name)
    return


def get_args(args, i):
    """
    split target files to multiprocess process args
    """
    out_args = dict()
    out_args['tmp_dir'] = args.tmp_dir
    main_process = i==0

    # fetch all pragent files
    files = open(os.path.join('data', args.path_id), 'r').readlines()
    xth, ytotal = list(map(lambda x: int(x),  args.subset.split('/')))
    assert xth < ytotal, f"xth should be smaller than ytotal: subset = {args.subset}"
    files_per_subset = len(files) // ytotal + 1
    files = files[xth * files_per_subset, min((xth+1) * files_per_subset, len(files))]
    if main_process:
        print("per subset files:", len(files))
    
    files_per_process = len(files) // args.process_num + 1
    if main_process:
        print("per process files:", files_per_process)

    # fetch file ids for current process
    process_files = files[i*files_per_process: min((i+1)*files_per_process, len(files))]
    out_args['files_to_download'] = ['https://data.commoncrawl.org/'+f.strip('\n') for f in process_files]
    f_id = [f.split('/')[-1].split('.')[0] for f in process_files]
    # if not os.path.exists(os.path.join(args.output_dir, args.path_id)):
    #     os.mkdir(os.path.join(args.output_dir, args.path_id))
    out_args['target_path'] = [os.path.join(args.output_dir, args.path_id, f+'.txt') for f in f_id]
    return (out_args['files_to_download'], out_args['tmp_dir'], out_args['target_path'], main_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Common Crawl Cleaner', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.process_num):
        Process(target=main, args=get_args(args, i)).start()
    
