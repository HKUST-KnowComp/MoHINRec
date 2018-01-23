#coding=utf8
'''
    preprocessing data for the experiments
'''
import numpy as np
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
#from gensim import corpora, models
#import gensim
import re
from str_util import unicode2str, str2unicode
from utils import save_triplets
import os
#from yelp_extractor import extract_from_json

#yelp_dir = 'data/yelp/'
#dir_ = 'data/yelp-200k/'
#dir_ = 'data/cikm-yelp/'
#dir_ = 'data/yelp-25k/'
dir_ = 'data/yelp-50k/'
#dir_ = 'data/yelp/'
#dir_ = 'data/movielens/'

def lda_pre():
    '''
        preprocessing texts for LDA model,
        remove numbers, tokenization, stopwords removal, stemming
        save processed text into db
    '''
    reviews = extract_from_json('review')
    id2raw, id2prew = {}, {}
    #lines = open(yelp_dir+filename, 'r').readlines()
    #parts = [l.strip().split('\t') for l in lines if not l.startswith('#')]
    #texts = [str2unicode(p[3]) for p in parts]

    tokenizer = RegexpTokenizer(r'\w+')
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # create English stop words list
    en_stop = get_stop_words('en')

    wfilename = 'amazon_raw_reviews.ldapre'
    fw = open(dir_+wfilename, 'w+')
    fw.write('#user_id\titem_id\treview_id\trate\tprocessed_text\n')
    for ind, r in enumerate(reviews):
        if (ind+1) % 100000 == 0:
            print 'processing %s/%s' % (ind+1,len(reviews))
        text = r[0]
        rid = r['review_id']
        rate = r['stars']
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
        r['processed_text'] = p_text
        parts = [unicode2str(rid), str(rate), unicode2str(p_text)]
        fw.write('%s\n' % '\t'.join(parts))
    fw.close()
    print 'finish preprocessing, res saved in %s' % wfilename

def get_user_average_rate():
    reviews = extract_from_json('review')
    uid2rates = {}
    for rev in reviews:
        uid = rev['user_id']
        rate = rev['stars']
        uid2rates.setdefault(uid,[]).append(rate)
    uid2rates = [(uid, sum(rates) * 1.0 / len(rates)) for uid, rates in uid2rates.items()]
    filename = 'user_avg_rate.txt'
    fw = open(yelp_dir+filename, 'w+')
    fw.write('\n'.join(['%s\t%s' % (uid,round(avg,2)) for uid,avg in uid2rates]))

def remove_cold_start_users():
    '''
        remove users whose rating number is less than 5
    '''
    filename = dir_ + 'ratings.txt'
    lines = open(filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    uid2num, iid2num = {},{}
    for u,i,r in parts:
        uid2num[u] = uid2num.get(u,0) + 1
    res = []
    for u,i,r in parts:
        if uid2num[u] >= 5:
            res.append('%s\t%s\t%s' % (u,i,r))
    wfilename = dir_ + 'ratings_filter5.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

def generate_rid_aspect_triplets():
    rid_map_filename = yelp_dir + 'rid_ent_map.txt'
    lines = open(rid_map_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    ent2rid = {str2unicode(r[1]):r[0] for r in parts}

    aid_map_filename = yelp_dir + 'aid_topic_map.txt'
    lines = open(aid_map_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    tid2aid = {r[1].replace('topic', ''):r[0] for r in parts}

    aspects_dir = yelp_dir  + 'aspects/lda_v1/'
    rid2aspects = {}
    #review_id rate topic_ids raw text
    tn = 10
    aspect_filename = 'review_topic%s.res' % tn
    type_ = 't%s_aspect' % tn
    lines = open(aspects_dir+aspect_filename,'r').readlines()
    parts = [l.strip().split() for l in lines if not l.startswith('#')]
    res = []
    for info in parts:
        try:
            ent = str2unicode(info[0])
            topic_str = info[2].strip()
            if not topic_str:
                continue
            #topic_ids = [{'topic%s': % t.split(',')[0], } for t in topic_str.split('|')]
            topic_ids = [t.split(',') for t in topic_str.split('|')]
            for k, v in topic_ids:
                res.append('%s\t%s\t%s' % (ent2rid[ent], tid2aid[k], v))
        except Exception as e:
            print e
            print topic_str
    wfilename = yelp_dir + 'rid_aid_weights.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

def generate_exp_split():
    '''
        generate 5 groups of splitting data, given the rating data
        80%-20% train-test ratio
    '''
    rating_filename = 'ratings'
    ratings = np.loadtxt(dir_ + rating_filename + '.txt', dtype=np.float64)
    for n in xrange(5):
        exp_dir = dir_ + 'exp_split/%s/' % (n+1)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
            print 'create dir %s' % exp_dir
        train_filename = dir_ + 'exp_split/%s/%s_train_%s.txt' % (n+1, rating_filename, n+1)
        test_filename = dir_ + 'exp_split/%s/%s_test_%s.txt' % (n+1, rating_filename, n+1)
        rand_inds = np.random.permutation(ratings.shape[0])
        train_num = int(ratings.shape[0] * 0.8)
        train_data = ratings[rand_inds[:train_num]]
        test_data = ratings[rand_inds[train_num:]]
        np.savetxt(train_filename, train_data[:,:3], '%d\t%d\t%.1f')
        np.savetxt(test_filename, test_data[:,:3], '%d\t%d\t%.1f')

def generate_con_ids():
    '''
        generate consective ids for uids and bids in ratings
        start from 1
    '''
    rating_filename = 'ratings.txt'
    lines = open(dir_+rating_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    uids = set([r[0] for r in parts])
    bids = set([r[1] for r in parts])
    uid2ind = {v:str(k+1) for k,v in enumerate(uids)}
    bid2ind = {v:str(k+1) for k,v in enumerate(bids)}
    parts = [(uid2ind[u], bid2ind[b],r) for u,b,r in parts]
    wfilename = 'ratings_conids.txt'
    save_triplets(dir_+wfilename, parts)

def generate_validation_set():
    '''
        based on existing test set, randomly split it into validation and test set by the ratio 1:1
    '''
    for n in xrange(5):
        exp_dir = dir_ + 'exp_split/%s/' % (n+1)
        test_filename = dir_ + 'exp_split/%s/ratings_test_%s.txt' % (n+1, n+1)
        val_filename = dir_ + 'exp_split/%s/val_%s.txt' % (n+1, n+1)
        ratings = np.loadtxt(test_filename)
        rand_inds = np.random.permutation(ratings.shape[0])
        val_num = int(ratings.shape[0] * 0.5)
        val_data = ratings[rand_inds[:val_num]]
        test_data = ratings[rand_inds[val_num:]]
        np.savetxt(val_filename, val_data[:,:3], '%d\t%d\t%.1f')
        np.savetxt(test_filename.replace('ratings_', ''), test_data[:,:3], '%d\t%d\t%.1f')

if __name__ == '__main__':
    #get_user_average_rate()
    #remove_cold_start_users()
    #generate_rid_aspect_triplets()
    #generate_exp_split()
    #generate_con_ids()
    generate_validation_set()
