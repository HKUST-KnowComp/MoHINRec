#coding=utf8
'''
    sample data of the large scale ratings
'''
import sys
import os
import numpy as np
import random

def sample_by_obs():
    filename = 'data/yelp/ratings_filter5.txt'
    ratings = np.loadtxt(filename)
    N = ratings.shape[0]
    inds = np.random.choice(N, 200000, replace=False)
    sample_ratings = ratings[inds]
    wfilename = 'data/yelp-200k/ratings.txt'
    np.savetxt(wfilename,sample_ratings,fmt='%d\t%d\t%.1f')

def sample_by_rows_cols():
    '''
        10k: /12
        100k: /4
        50k: /6
        5k: /17
    '''
    filename = 'data/yelp/ratings_filter5.txt'
    ratings = np.loadtxt(filename)
    uids = np.unique(ratings[:,0])
    bids = np.unique(ratings[:,1])
    un = uids.size
    bn = bids.size
    print un, bn
    inds = np.random.choice(un, un / 17, replace=False)
    s_uids = uids[inds]
    inds = np.random.choice(bn, bn / 17, replace=False)
    s_bids = bids[inds]

    s_uids, s_bids = set(s_uids), set(s_bids)
    sample_ratings = [r for r in ratings if r[0] in s_uids and r[1] in s_bids]
    s_uids = list(set([r[0] for r in sample_ratings]))
    s_bids = list(set([r[1] for r in sample_ratings]))

    wdir = 'data/yelp-5k/'
    if not os.path.isdir(wdir):
        os.makedirs(wdir)
        print 'create %s' % wdir
    wfilename = wdir + 'ratings.txt'
    uid_filename = wdir + 'uids.txt'
    bid_filename = wdir + 'bids.txt'
    np.savetxt(uid_filename,s_uids,fmt='%d')
    np.savetxt(bid_filename,s_bids,fmt='%d')
    np.savetxt(wfilename,sample_ratings,fmt='%d\t%d\t%.1f')

if sys.argv[1] == 'row-col':
    sample_by_rows_cols()
