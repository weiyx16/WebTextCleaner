"""
    We randomly split the data into from a large corpus to a small corpus.  [Only for jsonl.gz files, like OSCAR]
    python dataSplitJsonlgz.py --output-dir /output/OSCAR-2201/compressed/en_meta/ --data-size 5GB
"""

import os
import gzip
from tqdm import tqdm
import random
import jsonlines
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

    compress_ratio = 3.0  # gz to source (computed by one example)
    content_ratio = 0.77  # not all content are we wanted

    count = 0
    files_list = []
    for f in tqdm(files):
        fsize = os.path.getsize(os.path.join(args.output_dir, f))
        fsize_to_unit = fsizeInUnit(fsize, data_size_unit) * compress_ratio * content_ratio
        files_list.append(f)
        count += fsize_to_unit 
        if count > data_size:
            print(f" we reach {count} {data_size_unit} with {len(files_list)} files")
            break

    # merge datas
    with open('merged_files.txt', 'w') as fw:
        for f in tqdm(files_list):
            os.system(f'cp {os.path.join(args.output_dir, f)} ./')
            g_file = gzip.GzipFile(f)
            open(f[:-3], 'wb+').write(g_file.read())
            lines = list(jsonlines.open(f[:-3], 'r'))
            lines = [l['content'].strip()+'\n' for l in lines]
            fw.writelines(lines)
            fw.writelines(['\n'])
            g_file.close()
            os.system('rm -f '+f)
            os.system('rm -f '+f[:-3])