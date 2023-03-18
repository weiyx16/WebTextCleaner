import wget
import gzip
import os

url_list = [
    'crawl-data/CC-MAIN-2013-20/', 'crawl-data/CC-MAIN-2013-48/', 'crawl-data/CC-MAIN-2014-10/', 'crawl-data/CC-MAIN-2014-15/',
    'crawl-data/CC-MAIN-2014-23/', 'crawl-data/CC-MAIN-2014-35/', 'crawl-data/CC-MAIN-2014-41/', 'crawl-data/CC-MAIN-2014-42/',
    'crawl-data/CC-MAIN-2014-49/', 'crawl-data/CC-MAIN-2014-52/', 'crawl-data/CC-MAIN-2015-06/', 'crawl-data/CC-MAIN-2015-11/',
    'crawl-data/CC-MAIN-2015-14/', 'crawl-data/CC-MAIN-2015-18/', 'crawl-data/CC-MAIN-2015-22/', 'crawl-data/CC-MAIN-2015-27/',
    'crawl-data/CC-MAIN-2015-32/', 'crawl-data/CC-MAIN-2015-35/', 'crawl-data/CC-MAIN-2015-40/', 'crawl-data/CC-MAIN-2015-48/',
    'crawl-data/CC-MAIN-2016-07/', 'crawl-data/CC-MAIN-2016-18/', 'crawl-data/CC-MAIN-2016-22/', 'crawl-data/CC-MAIN-2016-26/',
    'crawl-data/CC-MAIN-2016-30/', 'crawl-data/CC-MAIN-2016-36/', 'crawl-data/CC-MAIN-2016-40/', 'crawl-data/CC-MAIN-2016-44/',
    'crawl-data/CC-MAIN-2016-50/', 'crawl-data/CC-MAIN-2017-04/', 'crawl-data/CC-MAIN-2017-09/', 'crawl-data/CC-MAIN-2017-13/',
    'crawl-data/CC-MAIN-2017-17/', 'crawl-data/CC-MAIN-2017-22/', 'crawl-data/CC-MAIN-2017-26/', 'crawl-data/CC-MAIN-2017-30/',
    'crawl-data/CC-MAIN-2017-34/', 'crawl-data/CC-MAIN-2017-39/', 'crawl-data/CC-MAIN-2017-43/', 'crawl-data/CC-MAIN-2017-47/',
    'crawl-data/CC-MAIN-2017-51/', 'crawl-data/CC-MAIN-2018-05/', 'crawl-data/CC-MAIN-2018-09/', 'crawl-data/CC-MAIN-2018-13/',
    'crawl-data/CC-MAIN-2018-17/', 'crawl-data/CC-MAIN-2018-22/', 'crawl-data/CC-MAIN-2018-26/', 'crawl-data/CC-MAIN-2018-30/',
    'crawl-data/CC-MAIN-2018-34/', 'crawl-data/CC-MAIN-2018-39/', 'crawl-data/CC-MAIN-2018-43/', 'crawl-data/CC-MAIN-2018-47/',
    'crawl-data/CC-MAIN-2018-51/', 'crawl-data/CC-MAIN-2019-04/', 'crawl-data/CC-MAIN-2019-09/', 'crawl-data/CC-MAIN-2019-13/', 
    'crawl-data/CC-MAIN-2019-18/', 'crawl-data/CC-MAIN-2019-22/', 'crawl-data/CC-MAIN-2019-26/', 'crawl-data/CC-MAIN-2019-30/',
    'crawl-data/CC-MAIN-2019-35/', 'crawl-data/CC-MAIN-2019-39/', 'crawl-data/CC-MAIN-2019-43/', 'crawl-data/CC-MAIN-2019-47/',
    'crawl-data/CC-MAIN-2019-51/', 'crawl-data/CC-MAIN-2020-05/', 'crawl-data/CC-MAIN-2020-10/', 'crawl-data/CC-MAIN-2020-16/',
    'crawl-data/CC-MAIN-2020-24/', 'crawl-data/CC-MAIN-2020-29/', 'crawl-data/CC-MAIN-2020-34/', 'crawl-data/CC-MAIN-2020-40/',
    'crawl-data/CC-MAIN-2020-45/', 'crawl-data/CC-MAIN-2020-50/', 'crawl-data/CC-MAIN-2021-04/', 'crawl-data/CC-MAIN-2021-10/',
    'crawl-data/CC-MAIN-2021-17/', 'crawl-data/CC-MAIN-2021-21/', 'crawl-data/CC-MAIN-2021-25/', 'crawl-data/CC-MAIN-2021-31/',
    'crawl-data/CC-MAIN-2021-39/', 'crawl-data/CC-MAIN-2021-43/', 'crawl-data/CC-MAIN-2021-49/', 'crawl-data/CC-MAIN-2022-05/',
    'crawl-data/CC-MAIN-2022-21/', 'crawl-data/CC-MAIN-2022-27/', 'crawl-data/CC-MAIN-2022-33/', 'crawl-data/CC-MAIN-2022-40/'
]

if __name__ == '__main__':
    print('This code is updated on 22nd, Oct, 2022. If you wish to get later releasted data, add urls into the list')
    for split in url_list:
        out = os.path.join('path',split.split('/')[1])
        files = 'https://data.commoncrawl.org/'+split+'wet.paths.gz'
        wget.download(files, out+'.gz')
        g_file = gzip.GzipFile(out +'.gz')
        open(out, 'wb+').write(g_file.read())
        g_file.close()
        os.system('rm -f '+out +'.gz')
