#coding=utf8
'''
    processing amazon electronics dataset
'''
import sys
import json
import re
import numpy as np
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from str_util import unicode2str, str2unicode

dir_ = 'data/amazon/'
review_filename = dir_ + 'reviews.json'
meta_filename = dir_ + 'meta.json'

uid_map_filename = dir_ + 'user_id_map.txt'
lines = open(uid_map_filename,'r').readlines()
parts = [l.strip().split(',') for l in lines]
user2id = {k:v for k,v in parts}

item_map_filename = dir_ + 'item_id_map.txt'
lines = open(item_map_filename,'r').readlines()
parts = [l.strip().split(',') for l in lines]
item2id = {k:v for k,v in parts}

def extract_raw_ratings():
    lines = open(review_filename).readlines()
    res = []
    ui = set()
    for l in lines:
        review = json.loads(l.strip())
        user = review['reviewerID']
        item = review['asin']
        rate = review['overall']
        uik = '%s-%s' % (user,item)
        if uik not in ui:
            ui.add(uik)
            res.append('%s\t%s\t%s' % (user, item, rate))
    print 'extract %s unique triplets from %s raw reviews' % (len(res), len(lines))
    wfilename = dir_ + 'raw_ratings.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print 'raw ratings saved in ', wfilename

def generate_ratings():
    filename = dir_ + 'raw_ratings.txt'
    lines = open(filename, 'r').readlines()
    res = []
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join((u2id[u],item2id[i],r)) for u,i,r in parts]
    wfilename = dir_ + 'ratings.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print 'ratings saved in ', wfilename

def extract_brands():

    lines = open(meta_filename).readlines()
    res = []
    brand2id = {}
    item2brand_ids = {}
    for ind, l in enumerate(lines):
        l = l.strip().replace('"','').replace('\'','').replace('&amp;','&')
        asin_p = r'asin:\s*(.+?),'
        item = re.search(asin_p, l).group(1)
        #import pdb;pdb.set_trace()
        brand_p = r'brand:\s*(.+?),'
        brand_exist = re.search(brand_p, l)
        if brand_exist:
            brand = brand_exist.group(1).strip()
            if 'related:' in brand:
                continue
            brand_id = brand2id.setdefault(brand, len(brand2id.keys()) + 1)
            item2brand_ids[item] = brand_id
    for item, iid in item2id.items():
        if item in item2brand_ids:
            res.append('%s\t%s' % (iid, item2brand_ids[item]))
    print 'get %s unique brands from %s meta data ' % (len(brand2id), len(lines))

    wfilename = dir_ + 'item_brand.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print 'item_brand saved in ', wfilename

    res = ['%s,%s' % (k,v) for k,v in brand2id.items()]
    wfilename = dir_ + 'brand_id_map.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print 'brand_id_map saved in ', wfilename

def extract_cats():
    lines = open(meta_filename).readlines()
    res = []
    cat2id = {}
    item2cat_ids = {}
    for ind, l in enumerate(lines):
        l = l.strip().replace('"','').replace('\'','')
        asin_p = r'asin:\s*(.+?),'
        item = re.search(asin_p, l).group(1)
        #import pdb;pdb.set_trace()
        cat_p = r'categories:\s*\[+(.+?)\]'
        cats = re.search(cat_p, l).group(1)
        for c in cats.split(','):
            c = c.strip()
            cat_id = cat2id.setdefault(c, len(cat2id.keys()) + 1)
            item2cat_ids.setdefault(item,[]).append(cat_id)
    for item, iid in item2id.items():
        for cid in item2cat_ids[item]:
            res.append('%s\t%s' % (iid, cid))
    print 'get %s unique cats ' % (len(cat2id))

    wfilename = dir_ + 'item_cat.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print 'item_cat saved in ', wfilename

    res = ['%s,%s' % (k,v) for k,v in cat2id.items()]
    wfilename = dir_ + 'cat_id_map.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print 'cat_id_map saved in ', wfilename

def lda_pre():
    '''
        preprocessing texts for LDA model,
        remove numbers, tokenization, stopwords removal, stemming
        save processed text into db
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # create English stop words list
    en_stop = get_stop_words('en')


    lines = open(review_filename).readlines()

    wfilename = 'amazon_raw_reviews.ldapre'
    fw = open(dir_+wfilename, 'w+')
    fw.write('#user_id\titem_id\treview_id\trate\tprocessed_text\n')

    for ind, l in enumerate(lines):
        if (ind+1) % 100000 == 0:
            print 'processing %s/%s' % (ind+1,len(lines))
        review = json.loads(l.strip())
        user = review['reviewerID']
        item = review['asin']
        rate = review['overall']
        text = review['reviewText']

        #remove numbers
        text = re.sub(r'\d+', '', text)
        #remove \n
        text = re.sub(r'\n', ' ', text)

        #text = str2unicode(text)

        # clean and tokenize document string
        tokenized_text = tokenizer.tokenize(text.lower())

        # remove stop words from tokens
        stopped_tokens = [i for i in tokenized_text if not i in en_stop]

        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        p_text = '\t'.join(stemmed_tokens)
        parts = [user2id[user], item2id[item], 'review_%s' % (ind+1), str(rate), unicode2str(p_text)]
        fw.write('%s\n' % '\t'.join(parts))
    fw.close()
    print 'finish preprocessing, res saved in %s' % wfilename


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'ratings':
            generate_ratings()
        if sys.argv[1] == 'brand':
            extract_brands()
        if sys.argv[1] == 'cat':
            extract_cats()
        if sys.argv[1] == 'lda_pre':
            lda_pre()
