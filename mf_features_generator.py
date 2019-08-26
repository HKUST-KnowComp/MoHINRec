#coding=utf8
'''
    generate MF features from the meta-structure similarity
'''

import sys
import time
import logging
import os
import numpy as np
#from numba import jit

from mf import MF_BGD as MF
from utils import reverse_map

from logging_util import init_logger

topK = 500

def run(path_str, comb='', K=10):
    if path_str in ['ratings_only']:
        use_topK = False
    else:
        use_topK = True

    sim_filename = dir_ + 'sim_res/path_count/%s.res' % path_str
    if path_str == 'ratings_only':
        sim_filename = dir_ + 'ratings.txt'
    if use_topK:
        sim_filename = dir_ + 'sim_res/path_count/%s_top%s.res' % (path_str, topK)
    if comb:
        sim_filename = dir_ + 'sim_res/path_count/combs/%s_%s_top%s.res' % (path_str, comb, topK)
    start_time = time.time()
    data = np.loadtxt(sim_filename)
    uids = set(data[:,0].flatten())
    bids = set(data[:,1].flatten())
    uid2ind = {int(v):k for k,v in enumerate(uids)}
    ind2uid = reverse_map(uid2ind)
    bid2ind = {int(v):k for k,v in enumerate(bids)}
    ind2bid = reverse_map(bid2ind)

    data[:,0] = [uid2ind[int(r)] for r in data[:,0]]
    data[:,1] = [bid2ind[int(r)] for r in data[:,1]]

    print 'finish load data from %s, cost %.2f seconds, users: %s, items=%s' % (sim_filename, time.time() - start_time, len(uids), len(bids))

    eps, lamb, iters = 10, 10, 500
    print 'start generate mf features, (K, eps, reg, iters) = (%s, %s, %s, %s)' % (K, eps, lamb, iters)
    mf = MF(data=data, train_data=data, test_data=[], K=K, eps=eps, lamb=lamb, max_iter=iters, call_logger=logger)
    U,V = mf.run()
    start_time = time.time()
    wfilename = dir_ + 'mf_features/path_count/%s_user.dat' % (path_str)
    rank_dir = dir_ + 'mf_features/path_count/ranks/%s/' % K
    if K != 10 and not os.path.isdir(rank_dir):
        os.makedirs(rank_dir)

    if use_topK:
        #wfilename = dir_ + 'mf_features/path_count/%s_top%s_user.dat' % (path_str, topK)
        wfilename = dir_ + 'mf_features/path_count/%s_top%s_user.dat' % (path_str, topK)
    else:
        wfilename = dir_ + 'mf_features/path_count/%s_user.dat' % (path_str)

    fw = open(wfilename, 'w+')
    res = []
    for ind, fs in enumerate(U):
        row = []
        row.append(ind2uid[ind])
        row.extend(fs.flatten())
        res.append('\t'.join([str(t) for t in row]))

    fw.write('\n'.join(res))
    fw.close()
    print 'User-Features: %s saved in %s, cost %.2f seconds' % (U.shape, wfilename, time.time() - start_time)

    start_time = time.time()
    wfilename = dir_ + 'mf_features/path_count/%s_item.dat' % (path_str)
    if use_topK:
        #wfilename = dir_ + 'mf_features/path_count/%s_top%s_item.dat' % (path_str, topK)
        wfilename = dir_ + 'mf_features/path_count/%s_top%s_item.dat' % (path_str, topK)
    else:
        wfilename = dir_ + 'mf_features/path_count/%s_item.dat' % (path_str)

    fw = open(wfilename, 'w+')
    res = []
    for ind, fs in enumerate(V):
        row = []
        row.append(ind2bid[ind])
        row.extend(fs.flatten())
        res.append('\t'.join([str(t) for t in row]))

    fw.write('\n'.join(res))
    fw.close()
    print 'Item-Features: %s  saved in %s, cost %.2f seconds' % (V.shape, wfilename, time.time() - start_time)

def run_all_epinions():
    #    run(path_str)
    for path_str in ['ratings_only']:
        run(path_str)
    for path_str in ['UUB_m1', 'UUB_m2', 'UUB_m3', 'UUB_m4','UUB_m5','UUB_m6','UUB_m7']:
        for n in range(11):
            alpha = n * 0.1
            path_str1 = '%s_%s' % (path_str, alpha)
            print 'run for ', path_str1
            run(path_str1)

if __name__ == '__main__':
    global dir_
    if len(sys.argv) == 3:
        dt = sys.argv[1]
        split_num = sys.argv[2]
        dir_ = 'data/%s/exp_split/%s/' % (dt, split_num)
        log_filename = 'log/%s_mf_feature_geneartion_split%s.log' % (dt, split_num)
        exp_id = int(time.time())
        logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)
        run_all_epinions()
    else:
        print 'please speficy the data and path_str'
        sys.exit(0)

