"""
    We randomly split the data into from a large corpus to a small corpus.  [Only for jsonl files]
    python dataSplitTxt.py --file ./PileCC_part1of2.dedup.ExactLen200.txt --data-size 5GB
"""

import os
import jsonlines
from tqdm import tqdm
import random
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('get information', add_help=False)
    parser.add_argument('--file', default='./test.txt', type = str)
    parser.add_argument('--data-size', default='5GB', type = str, help="target data size")
    return parser

def fsizeInUnit(fsize, unit):
    if unit == 'MB':
        return fsize / 1024 / 1024
    elif unit == 'GB':
        return fsize / 1024 / 1024 / 1024
    elif unit == 'TB':
        return fsize / 1024 / 1024 / 1024 / 1024


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Common Crawl Cleaner', parents=[get_args_parser()])
    args = parser.parse_args()
    data_size = float(args.data_size[:-2])
    data_size_unit = args.data_size[-2:]
    assert data_size_unit in ['MB', 'GB', 'TB']

    with open('merged_files.txt', 'w') as fw:
        fr = open(args.file, 'r')
        lines = fr.readlines()
        blocks = []
        tmp = ''
        for l in tqdm(lines):
            if l == '\n':
                if tmp.strip()=='':
                    tmp = ''
                    continue
                # blocks.append(Row(text=tmp))
                blocks.append(tmp)
                tmp = ''
            else:
                tmp = tmp+l.strip('\n')+'\n'
        random.shuffle(blocks)
        count = 0
        stat_freq = 1000
        for l in tqdm(blocks):
            texts = l
            fw.writelines(texts)
            fw.writelines(['\n'])
            count += 1
            if count % stat_freq == 0:
                fsize = os.path.getsize('merged_files.txt')
                fsize_to_unit = fsizeInUnit(fsize, data_size_unit)
                if fsize_to_unit > data_size:
                    print(f" we reach {fsize_to_unit} {data_size_unit} with {count} sequences")
                    break