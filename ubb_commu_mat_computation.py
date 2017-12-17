#coding=utf8
'''
    calculate the UB-*-B similarity for users and items in the whole dataset
    the similarity is the path_count, i.e. number of path instannces connecting user and biz
'''
import time
import threading
import cPickle as pickle
import json
import logging
import numpy as np
import sys
from utils import reverse_map, generate_adj_mat
from cal_commuting_mat import *

from logging_util import init_logger
logger = None
threads_finish = []

dir_ = 'data/yelp-200k/'

def get_topK_items(comm_res, ind2uid, ind2bid, topK=1000):
    start = time.time()
    U, _ = comm_res.shape
    triplets = []
    for i in xrange(U):
        #import pdb;pdb.set_trace()
        items = comm_res.getrow(i).toarray().flatten()
        cols = np.argpartition(-items, topK).flatten()[:topK]#descending order
        cols = [c for c in cols if items[c] > 0]#those less than 1000 non-zero entries, need to be removed zero ones
        triplets.extend([(ind2uid[i], ind2bid[c], items[c]) for c in cols])
    logger.info('get topK items, total %s entries, cost %.2f seconds', len(triplets), time.time() - start)
    return triplets

def get_triplets(comm_res, ind2row, ind2col):
    coo = comm_res.tocoo(copy=False)
    rows, cols, vs = coo.row, coo.col, coo.data
    triplets = []
    for r, c, v in zip(rows, cols, vs):
        triplets.append((ind2row[r], ind2col[c], v))
    print 'get %s triplets' % len(triplets)
    return triplets

def save_triplets(filename, triplets, is_append=False):
    if is_append:
        fw = open(filename, 'a+')
    else:
        fw = open(filename, 'w+')
    fw.write('\n'.join(['%s\t%s\t%s' % (h,t,v) for h,t,v in triplets]))
    fw.close()

def batch_save_comm_res(path_str, wfilename, comm_res, ind2row, ind2col, is_sample=True):
    coo = comm_res.tocoo(copy=False)
    step = 10000000
    N = len(coo.row) / step
    for i in range(N+1):
        start_time = time.time()
        triplets = []
        start = i * step
        end = start + step
        rows = coo.row[start:end]
        cols = coo.col[start:end]
        vs = coo.data[start:end]
        for r, c, v in zip(rows, cols, vs):
            triplets.append((ind2row[r], ind2col[c], v))
        save_triplets(wfilename, triplets, is_append=True)
        logger.info('finish saving 10M %s triplets in %s, progress: %s/%s, cost %.2f seconds', path_str, wfilename, (i+1) * step, len(coo.data), time.time() - start_time)

def cal_comm_mat_UBB(path_str, block_num, topK, uid_filename, ubp_filename, bid2ind, ind2bid, adj_bo, adj_bo_t):
    '''
        calculate the commuting matrix in U-B-*-B style
        in fact, only need to calculate BB
        the whole data
    '''
    global threads_finish
    logger.info('start cal %s, block %s, topK=%s', path_str, block_num, topK)
    lines = open(uid_filename, 'r').readlines()
    uids = [int(l.strip()) for l in lines]
    uid2ind = {v:k for k,v in enumerate(uids)}
    ind2uid = reverse_map(uid2ind)
    logger.info('run cal_comm_mat for %s uids in %s', len(uids), uid_filename)

    ubp = np.loadtxt(ubp_filename, dtype=np.int64)
    adj_ub, adj_ub_t = generate_adj_mat(ubp, uid2ind, bid2ind)

    t1 = time.time()
    comm_res = cal_mat_ubb(path_str, adj_ub, adj_bo, adj_bo_t)

    t2 = time.time()
    logger.info('cal res of %s cost %2.f seconds', path_str, t2 - t1)
    logger.info('comm_mat: shape=%s,density=%s', comm_res.shape, comm_res.nnz * 1.0/comm_res.shape[0]/comm_res.shape[1])
    wfilename = dir_ + 'commuting_mat/split_res_top%s/%s_%s.res' % (topK, path_str, block_num)
    #batch_save_comm_res(path_str, wfilename, comm_res, ind2uid, ind2bid, is_sample=False)
    triplets = get_topK_items(comm_res, ind2uid, ind2bid, topK)
    save_triplets(wfilename, triplets)
    t3 = time.time()
    logger.info('finish saving res of %s in %s, cost %2.f seconds', path_str, wfilename,  t3 - t2)
    ind = block_num % 3
    threads_finish[ind - 1] = True

def cal_comm_mat_BB(path_str, bid2ind, ind2bid, adj_bo, adj_bo_t):
    '''
        calculate the commuting matrix in B-*-B style
        in fact, only need to calculate BB
        the whole data
    '''
    t1 = time.time()
    comm_res = cal_mat_bb(path_str, adj_bo, adj_bo_t)
    t2 = time.time()
    logger.info('cal res of %s cost %2.f seconds', path_str, t2 - t1)
    logger.info('comm_mat: shape=%s,density=%s', comm_res.shape, comm_res.nnz * 1.0/comm_res.shape[0]/comm_res.shape[1])

    wfilename = dir_ + 'sim_res/path_count/%s.res' % (path_str)
    #triplets = get_triplets(comm_res, ind2bid, ind2bid)
    #save_triplets(wfilename, triplets)
    batch_save_comm_res(path_str, wfilename, comm_res, ind2bid, ind2bid, is_sample=False)
    t3 = time.time()
    logger.info('finish saving res of %s in %s, cost %2.f seconds', path_str, wfilename,  t3 - t2)

    #wfilename = dir_ + 'sim_res/path_count/%s_spa_mat.pickle' % path_str
    #fw = open(wfilename, 'w+')
    #json.dump(comm_res, fw)
    #map_filename = dir_ + 'sim_res/path_count/%s_spa_mat_id_map.pickle' % path_str
    #fw = open(map_filename, 'w+')
    #pickle.dump(ind2bid, fw, pickle.HIGHEST_PROTOCOL)
    #print 'finish saving sparse mat in ', wfilename

def get_bo(path_str, bid2ind):

    #U-pos-B-Cat-B
    if 'State' in path_str:
        sfilename = dir_ + 'bid_state.txt'
    elif 'Cat' in path_str:
        sfilename = dir_ + 'bid_cat.txt'
    elif 'City' in path_str:
        sfilename = dir_ + 'bid_city.txt'
    elif 'Star' in path_str:
        sfilename = dir_ + 'bid_stars.txt'

    lines = open(sfilename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    bos = [(int(b), int(o)) for b,o in parts]
    ond2ind = {v:k for k,v in enumerate(set([o for _, o in bos]))}
    ind2ond = reverse_map(ond2ind)
    adj_bo, adj_bo_t = generate_adj_mat(bos, bid2ind, ond2ind)
    return adj_bo, adj_bo_t

def load_bids(bid_filename):
    lines = open(bid_filename, 'r').readlines()
    bids = [int(l.strip()) for l in lines]
    bid2ind = {v:k for k,v in enumerate(bids)}
    ind2bid = reverse_map(bid2ind)
    logger.info('run cal_comm_mat for %s bids in %s', len(bids), bid_filename)
    return bid2ind, ind2bid

def run(path_str, group_num):
    global logger
    global threads_finish
    exp_id = int(time.time())
    start_time = time.time()
    topK = 1000
    print 'start, path_str=%s, group_num=%s, topK=%s' % (path_str, group_num, topK)

    log_filename = 'log/ubb_computation_%s' % path_str
    logger = init_logger('exp_%s' % str(exp_id), log_filename, logging.INFO, False)

    bid_filename = dir_ + 'bids.txt'

    bid2ind, ind2bid = load_bids(bid_filename)
    adj_bo, adj_bo_t = get_bo(path_str, bid2ind)

    threads = []
    threads_finish = [False] * 3
    for ind in range(1,4):
        ind = (group_num - 1) * 3 + ind
        uid_filename = dir_ + 'split_uids/uids_filter5_%s.txt' % ind
        ubp_filename = dir_ + 'split_uids/uid_pos_bid_filter5_%s.txt' % ind
        threads.append(threading.Thread(target=cal_comm_mat_UBB, args=(path_str, ind, topK, uid_filename, ubp_filename, bid2ind, ind2bid, adj_bo, adj_bo_t)))

    for t in threads:
        t.daemon = True
        t.start()

    while True:
        time.sleep(1)
        if sum(threads_finish) == 3:
            logger.info('all thread finished, path_str=%s, group_num=%s, topK=%s, total cose %.2f seconds', path_str, group_num, topK, time.time() - start_time)
            break

def run_single_thread(path_str):

    exp_id = int(time.time())
    start_time = time.time()
    top = 1000
    ind = -1

    log_filename = 'log/ubb_computation_%s' % path_str
    logger = init_logger('exp_%s' % str(exp_id), log_filename, logging.INFO, False)
    logger.info('start, path_str=%s, group_num=%s, topK=%s', path_str, group_num, topK)

    bid_filename = dir_ + 'bids.txt'
    uid_filename = dir_ + 'uids.txt'
    upb_filename = dir_ + 'uid_pos_neg.txt'

    bid2ind, ind2bid = load_bids(bid_filename)
    adj_bo, adj_bo_t = get_bo(path_str, bid2ind)

    cal_comm_mat_UBB(path_str, ind, topK, uid_filename, upb_filename, bid2ind, ind2bid, adj_bo, adj_bo_t)

def run_bb(path_str):

    global logger
    exp_id = int(time.time())
    log_filename = 'log/bb_computation_%s.log' % path_str
    logger = init_logger('exp_%s' % str(exp_id), log_filename, logging.INFO, False)

    bid_filename = dir_ + 'pos_bids.txt'
    bid2ind, ind2bid = load_bids(bid_filename)

    adj_bo, adj_bo_t = get_bo(path_str, bid2ind)
    cal_comm_mat_BB(path_str, bid2ind, ind2bid, adj_bo, adj_bo_t)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        path_str = sys.argv[1]
        group_num = int(sys.argv[2])
        run(path_str, group_num)
    if len(sys.argv) == 2:
        path_str = sys.argv[1]
        run_bb(path_str)

