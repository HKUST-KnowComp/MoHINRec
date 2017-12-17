#coding=utf8
'''
    calculate the user-item similarity based on different meta-structures
    this is for U-*-U-B style meta-graphs
'''

import sys
import time
import cPickle as pickle

import numpy as np
from scipy.sparse import csr_matrix as csr
from scipy.sparse import csc_matrix as csc

from utils import generate_adj_mat, reverse_map

dir_ = 'data/yelp/'

stru_str = 'URSPRUB'

if len(sys.argv) == 2:
    stru_str = sys.argv[1]
    print 'stru_str is specified: ', stru_str
else:
    print 'please speficy the stru_str'
    sys.exit(0)

filename = dir_ + 'uids_filter5.txt'
uids = [int(l.strip()) for l in open(filename, 'r').readlines()]
uids = set(uids)
uid2ind = {v:k for k,v in enumerate(uids)}
ind2uid = reverse_map(uid2ind)

filename = dir_ + 'bids.txt'
lines = open(filename, 'r').readlines()
bids = [int(l.strip()) for l in lines]
bids = list(set(bids))
bid2ind = {v:k for k,v in enumerate(bids)}
ind2bid = reverse_map(bid2ind)

def get_candidate_bids():
    filename = dir_ + 'sim_res/path_count/%s.res' % stru_str#path count of meta-structure instances
    lines = open(filename, 'r').readlines()
    uid2can_bids = {}
    for l in lines:
        parts = l.strip().split()
        uid2can_bids.setdefault(int(parts[0]), []).append(int(parts[1]))
    return uid2can_bids

t2 = time.time()
uid2can_bids = get_candidate_bids()
t3 = time.time()

ubp_filename = dir_ + 'filter5_uid_pos_bid.txt'
data2 = np.loadtxt(ubp_filename, dtype=int)
rows = [uid2ind[int(r)] for r in data2[:,0]]
cols = [bid2ind[int(r)] for r in data2[:,1]]
ubp_csc = csc(([1] * len(rows), (rows, cols)), shape=[len(uids), len(bids)])
del data2
t4 = time.time()
print 'finish creating ubp_csc: %s ,cost %.2f seconds' % (ubp_csc.shape, t4 -t3)

def cal_ursru(stru_str):

    if 'P' in stru_str:
        ur_filename = dir_ + 'filter5_uid_pos_rid.txt'
        rb_filename =  dir_ + 'filter5_rid_pos_bid.txt'

        ind2rid_filename = dir_ + 'sim_res/path_count/RAPR_spa_mat_id_map.pickle'
        rar_mat_filename = dir_ + 'sim_res/path_count/RAPR_spa_mat.pickle'
    elif 'N' in stru_str:
        ur_filename = dir_ + 'filter5_uid_neg_rid.txt'
        rb_filename =  dir_ + 'filter5_rid_neg_bid.txt'

        ind2rid_filename = dir_ + 'sim_res/path_count/RANR_spa_mat_id_map.pickle'
        rar_mat_filename = dir_ + 'sim_res/path_count/RANR_spa_mat.pickle'
    f = open(ind2rid_filename, 'r')

    ind2rid = pickle.load(f)
    rid2ind = reverse_map(ind2rid)

    #ub = np.loadtxt(ub_filename, dtype=np.int64)
    #adj_ub, adj_ub_t = generate_adj_mat(ub, uid2ind, bid2ind)

    ur = np.loadtxt(ur_filename, dtype=np.int64)
    adj_ur, adj_ur_t = generate_adj_mat(ur, uid2ind, rid2ind)

    rpb = np.loadtxt(rb_filename, dtype=np.int64)
    adj_rb, adj_rb_t = generate_adj_mat(rpb, rid2ind, bid2ind)

    start = time.time()
    RBR = adj_rb.dot(adj_rb_t)
    print 'RBR(%s), density=%.5f cost %.2f seconds' % (RBR.shape, RBR.nnz * 1.0/RBR.shape[0]/RBR.shape[1], time.time() - start)
    start = time.time()
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
    return URSRU

def cal_uu(stru_str, social_filename=''):
    '''
        calculate commuting matrix for U-*-U style
    '''
    upb_filename = dir_ + 'filter5_uid_pos_bid.txt'
    unb_filename = dir_ + 'filter5_uid_neg_bid.txt'
    social_filename = dir_ + 'filter5_user_social.txt'

    upb = np.loadtxt(upb_filename, dtype=np.int64)
    adj_upb, adj_upb_t = generate_adj_mat(upb, uid2ind, bid2ind)

    if stru_str == 'UBPUB':
        start = time.time()
        UBU = adj_upb.dot(adj_upb_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif stru_str == 'UBNUB':
        unb = np.loadtxt(unb_filename, dtype=np.int64)
        adj_unb, adj_unb_t = generate_adj_mat(unb, uid2ind, bid2ind)

        start = time.time()
        UBU = adj_unb.dot(adj_unb_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif stru_str == 'UUB':
        uu = np.loadtxt(social_filename, dtype=np.int64)
        adj_uu, adj_uu_t = generate_adj_mat(uu, uid2ind, uid2ind)

        start = time.time()
        UBU = adj_uu.dot(adj_uu_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)
    return UBU


if 'S' in stru_str:
    uu_csr = cal_ursru(stru_str)
else:
    uu_csr = cal_uu(stru_str)
t5 = time.time()
print 'finish cal USU: %s, cost %.2f seconds' % (uu_csr.shape, t5 -t4)

def cal_sim(uid, cal_uinds):
    sim = 0.0
    pn = 0.0
    ind1 = uid2ind[uid]
    cnt1 = uu_csr[ind1, ind1]
    for ind2 in cal_uinds:
        if uu_csr[ind1, ind2] > 0.0:
            pn += 1.0
            sim += 2.0 * uu_csr[ind1, ind2] / (cnt1 + uu_csr[ind2, ind2])#path sim
    nsim = sim / pn if pn else 0.0
    return sim, nsim

cnt = 0
#calculate similarity for every user, item pair
res, normalized_res = [], []
cal_start, start = time.time(), time.time()
for uc, uid in enumerate(uids):
    can_bids = uid2can_bids.get(uid, [])
    if not can_bids:
        continue
    for bc, bid in enumerate(can_bids):
        cnt += 1
        if cnt % 100000 == 0:
            end = time.time()
            print 'processing %s/%s user %s, %s/%s bid %s, cost %.2f seconds, total_cost %.2f minutes' % (uc, len(uids), uid, bc, len(can_bids), bid, end - start, (end - cal_start) / 60.0)
            start = end
        buids = set(ubp_csc[:,bid2ind[bid]].tocoo().row)
        sim, normalized_sim = cal_sim(uid, buids)
        if not sim:
            continue
        res.append((uid, bid, sim))
        normalized_res.append((uid, bid, normalized_sim))


def save(wfilename, res):
    fw = open(wfilename, 'w+')
    res = ['%s\t%s\t%s' % (u,i,s) for u,i,s in res]
    fw.write('\n'.join(res))
    fw.close()
    print 'finish saving %s pair of u-i(%s*%s) from %s, sim saved in %s' % (len(res), len(uids), len(bids), filename, wfilename)

print 'start saving sim res...'
t6 = time.time()
wfilename = dir_ + 'sim_res/path_sim/%s.res' % stru_str
save(wfilename, res)
wfilename = dir_ + 'sim_res/path_sim_norm/%s.res' % stru_str
save(wfilename, normalized_res)
t7 = time.time()
print 'finish saving two sim res and cal ends, cost %.2f seconds, total cost %.2f minutes' % (t7-t6, (t7-t2)/60.0)
