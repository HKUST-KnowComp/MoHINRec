#coding=utf8
'''
    calculate the similarity by commuting matrix baesd on meta-structure
'''
from numba import jit
import time
import sys
import os
import cPickle as pickle

import numpy as np
import bottleneck as bn
from scipy.sparse import csr_matrix as csr
from utils import reverse_map, generate_adj_mat, save_triplets

import copy

def get_topK_items(comm_res, ind2uid, ind2bid, topK=1000):
    start = time.time()
    U, _ = comm_res.shape
    triplets = []
    for i in xrange(U):
        items = comm_res.getrow(i).toarray().flatten()
        cols = np.argpartition(-items, topK).flatten()[:topK]#descending order
        cols = [c for c in cols if items[c] > 0]#those less than 1000 non-zero entries, need to be removed zero ones
        triplets.extend([(ind2uid[i], ind2bid[c], items[c]) for c in cols])
    #logger.info('get topK items, total %s entries, cost %.2f seconds', len(triplets), time.time() - start)
    print 'get top %s items, total %s entries, cost %.2f seconds' % (topK, len(triplets), time.time() - start)
    return triplets

def load_eids(eid_filename, type_):
    lines = open(eid_filename, 'r').readlines()
    eids = [int(l.strip()) for l in lines]
    eid2ind = {v:k for k,v in enumerate(eids)}
    ind2eid = reverse_map(eid2ind)
    #logger.info('get %s %s from %s', len(eids), type_, eid_filename)
    print 'get %s %s from %s' %(len(eids), type_, eid_filename)
    return eids, eid2ind, ind2eid

def compute_motif_matrix(adj_uu, adj_uu_t, path_str):
    B = adj_uu.multiply(adj_uu_t)
    U = adj_uu - B
    motif_matrix = None

    start = time.time()
    if path_str[-2:]=='m1':
        C = U.dot(U).multiply(U.T)
        motif_matrix = C + C.T

    elif path_str[-2:] == 'm2':
        C = B.dot(U).multiply(U.T) + U.dot(B).multiply(U.T) + U.dot(U).multiply(B)
        motif_matrix = C + C.T

    elif path_str[-2:] == 'm3':
        C = B.dot(B).multiply(U) + B.dot(U).multiply(B) + U.dot(B).multiply(B)
        motif_matrix = C + C.T

    elif path_str[-2:] == 'm4':
        motif_matrix = B.dot(B).multiply(B)

    elif path_str[-2:] == 'm5':
        C = U.dot(U).multiply(U) + U.dot(U.T).multiply(U) + U.T.dot(U).multiply(U)
        motif_matrix = C + C.T

    elif path_str[-2:] == 'm6':
        motif_matrix = U.dot(B).multiply(U) + B.dot(U.T).multiply(U.T) + U.T.dot(U).multiply(B)

    elif path_str[-2:] == 'm7':
        motif_matrix = U.T.dot(B).multiply(U.T) + B.dot(U).multiply(U) + U.dot(U.T).multiply(B)
    return motif_matrix

def cal_comm_mat_sm(path_str):
    '''
        calculate commuting matrix for U-*-U-pos-B style in merge way with 7 simple motifs (sm)
    '''
    uid_filename = dir_ + 'uids.txt'
    bid_filename = dir_ + 'bids.txt'
    ub_filename = dir_ + 'uid_bid.txt'

    print 'cal commut mat with motif for %s, filenames: %s, %s, %s' % (path_str, uid_filename, bid_filename, ub_filename)
    uids, uid2ind, ind2uid = load_eids(uid_filename, 'user')
    bids, bid2ind, ind2bid = load_eids(bid_filename, 'biz')

    # upb = np.loadtxt(upb_filename, dtype=np.int64)
    ub = np.loadtxt(ub_filename, dtype=np.int64)

    # adj_upb, adj_upb_t = generate_adj_mat(upb, uid2ind, bid2ind)
    adj_ub, adj_ub_t = generate_adj_mat(ub, uid2ind, bid2ind)

    social_filename = dir_ + 'user_social.txt'
    uu = np.loadtxt(social_filename, dtype=np.int64)
    adj_uu, adj_uu_t = generate_adj_mat(uu, uid2ind, uid2ind)

    motif_matrix = compute_motif_matrix(adj_uu, adj_uu_t, path_str)

    if path_str[:3] == 'UUB':
        base_matrix = adj_uu

    if path_str[:4] == 'UBUB':
        base_matrix = adj_ub.dot(adj_ub_t)

    #for n in range(1, 10):
    for n in range(11):
        alpha = n * 0.1
        UBU_merge = (1 - alpha) * base_matrix + alpha * motif_matrix
        start = time.time()
        UBUB = UBU_merge.dot(adj_ub)
        print 'UBUB(%s), density=%.5f cost %.2f seconds' % (UBUB.shape, UBUB.nnz * 1.0/UBUB.shape[0]/UBUB.shape[1], time.time() - start)
        start = time.time()
        K = 500

        #normal way
        triplets = get_topK_items(UBUB, ind2uid, ind2bid, topK=K)
        wfilename = dir_ + 'sim_res/path_count/%s_%s_top%s.res' % (path_str, alpha, K)
        save_triplets(wfilename, triplets)
        print 'finish saving %s %s entries in %s, cost %.2f seconds' % (len(triplets), path_str, wfilename, time.time() - start)

def cal_yelp_merge(split_num, dt):
    global dir_
    dir_ = 'data/%s/exp_split/%s/' % (dt, split_num)

    motif_paths = ['UUB_m%s' % r for r in range(1,8)]
    for path_str in motif_paths:
        cal_comm_mat_sm(path_str)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        dt = sys.argv[1]
        split_num = int(sys.argv[2])
        cal_yelp_merge(split_num,dt)
