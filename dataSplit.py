"""
    We randomly split the data into from a large corpus to a small corpus.
    python dataSplit.py --output-dir /output/common_crawl_cleaned/202050_v20230318_subset0of4/en/CC-MAIN-2020-50/ --data-size 5GB
"""

import os
from tqdm import tqdm
import random
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('get information', add_help=False)
    parser.add_argument('--output-dir', default='./output', type = str)
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
    files = os.listdir(args.output_dir)
    random.shuffle(files)

    data_size = float(args.data_size[:-2])
    data_size_unit = args.data_size[-2:]
    assert data_size_unit in ['MB', 'GB', 'TB']

    count = 0
    files_list = []
    for f in tqdm(files):
        fsize = os.path.getsize(os.path.join(args.output_dir, f))
        fsize_to_unit = fsizeInUnit(fsize, data_size_unit)
        files_list.append(f)
        count += fsize_to_unit 
        if count > data_size:
            print(f" we reach {count} {data_size_unit} with {len(files_list)} files")
            break

    # merge datas
    with open('merged_files.txt', 'w') as fw:
        for f in tqdm(files_list):
            fr = open(os.path.join(args.output_dir, f), 'r')
            lines = fr.readlines()
            fw.writelines(lines)
            fw.writelines(['\n'])

    