#coding=utf8
'''
    extract aspects from reviews
'''
import sys
import time

import numpy as np
from lda import LDA
from yelp_extractor import extract_from_json
from str_util import str2unicode, unicode2str

#yelp_dir = 'data/yelp/'
dir_ = 'data/amazon/'

def extract_review_texts():
    reviews = extract_from_json('review')
    raw_texts = []
    for r in reviews:
        raw_texts.append(r.get('text', ''))
    return raw_texts

def extract_business_info():
    bzs = extract_from_json('business')
    cities = set()
    for bz in bzs:
        cities.add((bz.get('state'), bz.get('city')))
    print cities

def load_reviews(filename):
    '''
        load review texts from file
        comment flag is "#"
        'user\titem\review_id\trate\ttext'
    '''
    lines = open(dir_+filename,'r').readlines()
    parts = [str2unicode(l.strip()).split() for l in lines if not l.startswith('#')]
    return parts

def extract_aspects_from_reviews(K):
    raw_review_filename = 'raw_reviews.txt.ldapre'
    raw_texts = load_reviews(raw_review_filename)
    lda_model = LDA(K=K, doc_set=raw_texts)
    lda_model.train()
    lda_model.save(yelp_dir + 'review_t%s.lda' % K)

def extract_aspects_for_reviews_v1(topic_num=10):
    '''
        load model from file, then obtain topics for every reivew
        some terms may not be in the model.id2term, ignore this in this version
    '''

    raw_review_filename = 'raw_reviews.ldapre'
    raw_texts = load_reviews(raw_review_filename)

    #train LDA
    model = LDA(K=topic_num, doc_set=raw_texts)
    model.train()
    model.save(dir_ + 'review_t%s.lda' % K)

    res_dir = dir_ + 'aspects/'
    res_filename = 'review_topic%s.res' % topic_num
    fw = open(res_dir+res_filename, 'w+')
    fw.write('#user\titem\treview_id\trate\ttopic_res\traw_text\n')
    model_topic_filename = 'topic%s.res' % topic_num
    tn = 0.0

    start = time.time()
    for ind, t in enumerate(raw_texts):
        user, item, rid = t[0], t[1], t[2]
        rate = float(t[3])
        terms = t[4:]
        topic_ids = model.get_document_topics(terms)
        tn += len(topic_ids)

        topic_str = '|'.join(['%s,%s' % (t,round(p,4)) for t,p in topic_ids])
        line = '%s\t%s\t%s\t%s\t%s\t%s' % (user, item, unicode2str(rid),rate,unicode2str(topic_str),'\t'.join([unicode2str(t) for t in terms]))
        fw.write(line + '\n')
        if (ind+1) % 100000 == 0:
            print 'cost  %.1fmin in this round, processed %s review:\n%s\n' % ((time.time() - start) / 60.0, ind+1, line)
            start = time.time()
    fw.close()

    topics_res = model.print_topics(topic_num)
    fw = open(res_dir+model_topic_filename, 'w+')
    fw.write('\n'.join(['%s,%s' % (t,unicode2str(r)) for t, r in topics_res]))
    fw.close()
    print 'finish extracting aspects for %s reviews(avg=%s), saved in %s, corpurs topics in %s' % (len(raw_texts), tn / len(raw_texts), res_filename, model_topic_filename)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        K = int(sys.argv[1])
        extract_aspects_for_reviews_v1(topic_num=K)

