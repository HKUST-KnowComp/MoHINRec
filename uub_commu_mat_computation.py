#coding=utf8
'''
    calculate the U-*-UB similarity for users and items in the whole dataset, especially the mata-structure based
'''
import time
import threading
import logging
import numpy as np
import sys
import cPickle as pickle

import bottleneck as bn
from scipy.sparse import csr_matrix as csr

from utils import reverse_map, generate_adj_mat
from cal_commuting_mat import *

from logging_util import init_logger
logger = None
threads_finish = []

dir_ = 'data/yelp/'

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

def cal_commt_mat_stru(path_str, uid_filename, rid_filename, bid_filename, aid_filename, ub_filename, ur_filename, rb_filename):
    '''
        Given meta_structure_str, generate the commuting matrix
        e.g. 'user-review-business,t10_aspect-review-user'
    '''

    uids, uid2ind, ind2uid = load_eids(uid_filename, 'user')
    #rids, rid2ind, ind2rid = load_eids(rid_filename, 'review')
    bids, bid2ind, ind2bid = load_eids(bid_filename, 'biz')
    aids, aid2ind, ind2aid = load_eids(aid_filename, 'aspect')

    if 'P' in path_str:
        ind2rid_filename = dir_ + 'sim_res/path_count/RAPR_spa_mat_id_map.pickle'
        rar_mat_filename = dir_ + 'sim_res/path_count/RAPR_spa_mat.pickle'
    elif 'N' in path_str:
        ind2rid_filename = dir_ + 'sim_res/path_count/RANR_spa_mat_id_map.pickle'
        rar_mat_filename = dir_ + 'sim_res/path_count/RANR_spa_mat.pickle'
    f = open(ind2rid_filename, 'r')

    ind2rid = pickle.load(f)

    rid2ind = reverse_map(ind2rid)

    ub = np.loadtxt(ub_filename, dtype=np.int64)
    adj_ub, adj_ub_t = generate_adj_mat(ub, uid2ind, bid2ind)

    ur = np.loadtxt(ur_filename, dtype=np.int64)
    adj_ur, adj_ur_t = generate_adj_mat(ur, uid2ind, rid2ind)

    rpb = np.loadtxt(rb_filename, dtype=np.int64)
    adj_rb, adj_rb_t = generate_adj_mat(rpb, rid2ind, bid2ind)

    start = time.time()
    RBR = adj_rb.dot(adj_rb_t)
    print 'RBR(%s), density=%.5f cost %.2f seconds' % (RBR.shape, RBR.nnz * 1.0/RBR.shape[0]/RBR.shape[1], time.time() - start)
    start = time.time()
    #RAR = adj_ra.dot(adj_ra_t)

    f = open(rar_mat_filename, 'r')
    RAR = pickle.load(f)
    print 'load RAR(%s), density=%.5f cost %.2f seconds' % (RAR.shape, RAR.nnz * 1.0/RAR.shape[0]/RAR.shape[1], time.time() - start)
    start = time.time()
    RSR = RBR.multiply(RAR)
    print 'RSR(%s), density=%.5f cost %.2f seconds' % (RSR.shape, RSR.nnz * 1.0/RSR.shape[0]/RSR.shape[1], time.time() - start)
    start = time.time()
    URSR = adj_ur.dot(RSR)
    print 'URSR(%s), density=%.5f cost %.2f seconds' % (URSR.shape, URSR.nnz * 1.0/URSR.shape[0]/URSR.shape[1], time.time() - start)
    start = time.time()
    URSRU = URSR.dot(adj_ur_t)
    print 'URSRU(%s), density=%.5f cost %.2f seconds' % (URSRU.shape, URSRU.nnz * 1.0/URSRU.shape[0]/URSRU.shape[1], time.time() - start)

    start = time.time()
    URSRUB = URSRU.dot(adj_ub)
    print 'URSRUB(%s), density=%.5f cost %.2f seconds' % (URSRUB.shape, URSRUB.nnz * 1.0/URSRUB.shape[0]/URSRUB.shape[1], time.time() - start)
    wfilename = dir_ + 'sim_res/path_count/%s.res' % path_str
    triplets = get_triplets(URSRUB, ind2uid, ind2bid)
    start = time.time()
    save_triplets(wfilename, triplets)
    print 'finish saving %s %s entries in %s, cost %.2f seconds' % (len(triplets), path_str, wfilename, time.time() - start)

def cal_commt_mat_uub(path_str, uid_filename, bid_filename, upb_filename, unb_filename='', social_filename=''):
    '''
        calculate commuting matrix for U-*-U-pos-B style
    '''
    print 'cal commut mat for %s, filenames: %s, %s, %s' % (path_str, uid_filename, bid_filename, upb_filename)
    uids, uid2ind, ind2uid = load_eids(uid_filename, 'user')
    bids, bid2ind, ind2bid = load_eids(bid_filename, 'biz')

    upb = np.loadtxt(upb_filename, dtype=np.int64)
    adj_upb, adj_upb_t = generate_adj_mat(upb, uid2ind, bid2ind)

    if path_str == 'UBPUB':
        start = time.time()
        UBU = adj_upb.dot(adj_upb_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str == 'UBNUB':
        unb = np.loadtxt(unb_filename, dtype=np.int64)
        adj_unb, adj_unb_t = generate_adj_mat(unb, uid2ind, bid2ind)

        start = time.time()
        UBU = adj_unb.dot(adj_unb_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str == 'UUB':
        uu = np.loadtxt(social_filename, dtype=np.int64)
        adj_uu, adj_uu_t = generate_adj_mat(uu, uid2ind, uid2ind)

        start = time.time()
        UBU = adj_uu.dot(adj_uu_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    start = time.time()
    UBUB = UBU.dot(adj_upb)
    print 'UBUB(%s), density=%.5f cost %.2f seconds' % (UBUB.shape, UBUB.nnz * 1.0/UBUB.shape[0]/UBUB.shape[1], time.time() - start)
    wfilename = dir_ + 'sim_res/path_count/%s.res' % path_str
    #triplets = get_triplets(UBUB, ind2uid, ind2bid)
    triplets = get_topK_items(UBUB, ind2uid, ind2bid)
    start = time.time()
    save_triplets(wfilename, triplets)
    print 'finish saving %s %s entries in %s, cost %.2f seconds' % (len(triplets), path_str, wfilename, time.time() - start)

def cal_rar(path_str, aid_filename, rid_filename, rpa_filename='', rna_filename=''):

    aids = open(aid_filename, 'r').readlines()
    aids = [int(r.strip()) for r in aids]
    aid2ind = {a:ind for ind, a in enumerate(aids)}#global ind
    ind2aid = reverse_map(aid2ind)

    rids = open(rid_filename, 'r').readlines()
    rids = [int(r.strip()) for r in rids]
    rid2ind = {r:ind for ind, r in enumerate(rids)}#global ind
    ind2rid = reverse_map(rid2ind)

    if path_str == 'RAPR':
        lines = open(rpa_filename, 'r').readlines()
    elif path_str == 'RANR':
        lines = open(rna_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    rpa = [(int(r), int(a), float(w)) for r, a, w in parts]
    adj_ra, adj_rat = generate_adj_mat(rpa, rid2ind, aid2ind, is_weight=True)

    #generate review aspects weights, we can do it from filename

    t1 = time.time()
    RA = adj_ra.toarray()
    t2 = time.time()
    print 'to dense RA%s cost %.2f seconds' % (RA.shape, t2 - t1)
    RAR_csr = cal_rar_block(RA, len(rid2ind), ind2rid, step=20000)
    print 'finish cal rar by blocks, cost %.2f minutes' % ((time.time() - t2) / 60.0)
    try:
        wfilename = dir_ + 'sim_res/path_count/%s_spa_mat.pickle' % path_str
        fw = open(wfilename, 'w+')
        pickle.dump(RAR_csr, fw, pickle.HIGHEST_PROTOCOL)
        map_filename = dir_ + 'sim_res/path_count/%s_spa_mat_id_map.pickle' % path_str
        fw = open(map_filename, 'w+')
        pickle.dump(ind2rid, fw, pickle.HIGHEST_PROTOCOL)
        print 'finish saving sparse mat in ', wfilename
    except Exception as e:
        print e

DEBUG = False
def cal_rar_block(RA, nR, ind2rid, step=10000, topK=100):
    if DEBUG:
        RA = np.random.rand(1005,10)
        ind2rid = {k:k for k in range(1005)}
        nR = 1005
        step, topK = 20, 10
        debug_RR = np.dot(RA, RA.T)
        col_inds = bn.argpartsort(-debug_RR, topK, axis=1)[:,:topK]
        dr,dc = col_inds.shape
        row_inds = np.tile(np.arange(dr).reshape(dr,1), dc)
        debug_res = np.zeros((1005,1005))
        debug_res[row_inds, col_inds] = 1

    step_num = RA.shape[0] / step
    data, rows, cols = [],[],[]
    rar_start = time.time()
    for i in range(step_num+1):
        r = i * step
        rblock = RA[r:r+step]

        b_top100_res = []
        b_top100_inds = []
        tmp_res = {}
        #finish 10000 users
        block_start = time.time()
        for j in range(step_num+1):
            c = j * step
            cblock = RA[c:c+step]
            t3 = time.time()

            dot_res = np.dot(rblock, cblock.T)# dot res: 10000 * 10000

            drc = dot_res.shape[1]
            tmp_topK = topK if topK < drc else drc

            top100_inds = bn.argpartsort(-dot_res, tmp_topK, axis=1)[:,:tmp_topK]#10000 * 100,100 indices of the top K weights, column indices in dot_res
            br, bc = top100_inds.shape
            top100_rows = np.tile(np.arange(br).reshape(br,1), bc)#number of colums = colums of top100 inds, usually =100

            top100_res = dot_res[top100_rows, top100_inds]#only need to preserve top 100 weights for global comparing

            b_top100_res.append(top100_res)

            b_top100_inds.append(top100_inds + c)#preserve the global indices, indices need to add the starting value of every block

        block_end = time.time()
        print 'finish calculating %s-th/%s block(%s*%s), cost %.2f seconds, rar_block cost %.2f minutes' % (i+1, step_num, step, step, block_end - block_start, (block_end - rar_start) / 60.0)
        b_top100_inds = np.concatenate(b_top100_inds, axis=1)
        b_top100_res = np.concatenate(b_top100_res, axis=1)

        top100_inds = bn.argpartsort(-b_top100_res, topK, axis=1)[:,:topK]#10000 * 100,100 indices of the top K weights
        tr, tc = top100_inds.shape
        #it may exists that not all 100 weights are zero, prob is very small, processing later
        top100_rows = np.tile(np.arange(tr).reshape(tr,1), tc)

        #global row and col inds are needed for the constructing the sparse matrix for RAR 
        top100_res = b_top100_res[top100_rows, top100_inds]#10000 * 100, some may equal zero
        b_col_top100_inds = b_top100_inds[top100_rows, top100_inds]#global column inds for top100, then we need to get global row inds

        #the following code is used for gurantee all the weights > 0.0, remove 0 weights, very uncommon
        trows, tcols = np.where(top100_res > 0.0)#return all the rows and cols of top100_res
        global_col_inds = b_col_top100_inds[trows, tcols]#value corresponded to the trows + r
        global_row_inds = trows + i * step
        rows.extend(global_row_inds)
        cols.extend(global_col_inds)

        triplets = []
        save_start = time.time()
        print 'finish selecting top %s for block %s, cost %.2f seconds, rar_block cost %.2f minutes' % (topK, i+1, save_start - block_end, (save_start - rar_start) / 60.0)
        #for r, c in zip(global_row_inds, global_col_inds):
        #    triplets.append((ind2rid[r], ind2rid[c], 1))
        #filename = dir_ + 'sim_res/path_count/%s_block_res/%s.dat' % (i+1)
        #save_triplets(filename, triplets)
        #save_end = time.time()
        #print 'finish processing block %s, res saved in %s, %s triplets, cost detail(total/compute/select/save): %.2f/%.2f/%.2f/%.2f seconds ' % (i+1, filename, len(triplets), save_end - block_start, block_end - block_start, save_start - block_end, save_end - save_start)

    data = np.ones(len(rows))#use one to replace the weights

    t4 = time.time()
    RAR_csr = csr((data, (rows, cols)), shape=[nR, nR])
    t5 = time.time()
    #print '10000 res to sparse matrix(%s) cost %.2f seconds' % (RAR_csr.shape, t5 - t4)
    if DEBUG:
        test = RAR_csr.toarray()
        test_res = (test == debug_res)
        if test_res.sum() == test.size:
            print '!!!block matrix equals directly dot matrix, the res is correct!!!'
        else:
            print 'two matrices are not equal, the res is wrong!!!'
    # make it symmetryic
    RAR = 0.5 * (RAR_csr + RAR_csr.transpose())
    RAR = RAR.ceil()#0.5 to 1, 1 to 1
    if DEBUG:
        import pdb;pdb.set_trace()
    return RAR

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

def load_eids(eid_filename, type_):
    lines = open(eid_filename, 'r').readlines()
    eids = [int(l.strip()) for l in lines]
    eid2ind = {v:k for k,v in enumerate(eids)}
    ind2eid = reverse_map(eid2ind)
    #logger.info('get %s %s from %s', len(eids), type_, eid_filename)
    print 'get %s %s from %s' %(len(eids), type_, eid_filename)
    return eids, eid2ind, ind2eid

def run(path_str):
    uid_filename = dir_ + 'uids_filter5.txt'
    rid_filename = dir_ + 'filter5_rids.txt'
    bid_filename = dir_ + 'bids.txt'
    aid_filename = dir_ + 'aids.txt'

    upb_filename = dir_ + 'filter5_uid_pos_bid.txt'
    unb_filename = dir_ + 'filter5_uid_neg_bid.txt'
    ur_filename = dir_ + 'filter5_uid_rid.txt'
    rpa_filename = dir_ + 'filter5_rid_pos_aid_weights.txt'
    rna_filename = dir_ + 'filter5_rid_neg_aid_weights.txt'

    if path_str == 'URSPRUB':
        rid_filename = dir_ + 'filter5_pos_rids.txt'
        ur_filename = dir_ + 'filter5_uid_pos_rid.txt'
        rb_filename =  dir_ + 'filter5_rid_pos_bid.txt'
        cal_commt_mat_stru(path_str, uid_filename, rid_filename, bid_filename, aid_filename, upb_filename, ur_filename, rb_filename)
    elif path_str == 'URSNRUB':
        rid_filename = dir_ + 'filter5_neg_rids.txt'
        ur_filename = dir_ + 'filter5_uid_neg_rid.txt'
        rb_filename =  dir_ + 'filter5_rid_neg_bid.txt'
        cal_commt_mat_stru(path_str, uid_filename, rid_filename, bid_filename, aid_filename, upb_filename, ur_filename, rb_filename)
    elif path_str == 'UBPUB':
        cal_commt_mat_uub(path_str, uid_filename, bid_filename, upb_filename)
    elif path_str == 'UBNUB':
        cal_commt_mat_uub(path_str, uid_filename, bid_filename, upb_filename, unb_filename)
    elif path_str == 'UUB':
        social_filename = dir_ + 'filter5_user_social.txt'
        cal_commt_mat_uub(path_str, uid_filename, bid_filename, upb_filename, unb_filename, social_filename)
    elif path_str == 'RAPR':
        rid_filename = dir_ + 'filter5_pos_rids.txt'
        cal_rar(path_str, aid_filename, rid_filename, rpa_filename=rpa_filename)
    elif path_str == 'RANR':
        rid_filename = dir_ + 'filter5_neg_rids.txt'
        cal_rar(path_str, aid_filename, rid_filename, rna_filename=rna_filename)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path_str = sys.argv[1]
        run(path_str)
