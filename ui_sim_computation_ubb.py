#coding=utf8
'''
    calculate the user-item similarity based on path_sim
    this script is for the U-pos-B-*-B style meta-path currently
'''

import sys
import time
import numpy as np
from scipy.sparse import csr_matrix as csr
from scipy.sparse import csc_matrix as csc

from utils import reverse_map, generate_adj_mat
from cal_commuting_mat import cal_mat_bb

stru_str = 'BCatB'
run_start = time.time()
if len(sys.argv) == 2:
    stru_str = sys.argv[1]
    print 'stru_str is specified: ', stru_str
else:
    print 'please speficy the stru_str'
    sys.exit(0)

dir_ = 'data/yelp/'

filename = dir_ + 'uids_filter5.txt'
uids = [int(l.strip()) for l in open(filename, 'r').readlines()]
uids = set(uids)
uid2ind = {v:k for k,v in enumerate(uids)}
ind2uid = reverse_map(uid2ind)

filename = dir_ + 'pos_bids.txt'
lines = open(filename, 'r').readlines()
bids = [int(l.strip()) for l in lines]
bids = list(set(bids))
bid2ind = {v:k for k,v in enumerate(bids)}
ind2bid = reverse_map(bid2ind)

def get_candidate_bids():
    filename = dir_ + 'sim_res/path_count/%s.res' % stru_str#path count of UBB instances
    lines = open(filename, 'r').readlines()
    uid2can_bids = {}
    for l in lines:
        parts = l.strip().split()
        uid2can_bids.setdefault(int(parts[0]), []).append(int(parts[1]))
    return uid2can_bids

t1 = time.time()
uid2can_bids = get_candidate_bids()
t2 = time.time()
print 'finish loading candidate bids, cost %.2f seconds' % (t2 - t1)

def get_bo(path_str, bid2ind):

    #U-pos-B-Cat-B
    if 'State' in path_str:
        sfilename = dir_ + 'pos_bid_state.txt'
    elif 'Cat' in path_str:
        sfilename = dir_ + 'pos_bid_cat.txt'
    elif 'City' in path_str:
        sfilename = dir_ + 'pos_bid_city.txt'
    elif 'Star' in path_str:
        sfilename = dir_ + 'pos_bid_stars.txt'

    lines = open(sfilename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    bos = [(int(b), int(o)) for b,o in parts]
    ond2ind = {v:k for k,v in enumerate(set([o for _, o in bos]))}
    ind2ond = reverse_map(ond2ind)
    adj_bo, adj_bo_t = generate_adj_mat(bos, bid2ind, ond2ind)
    return adj_bo, adj_bo_t

adj_bo, adj_bo_t = get_bo(stru_str, bid2ind)
bb_csr = cal_mat_bb(stru_str, adj_bo, adj_bo_t)
t3 = time.time()
print 'finish creating bb_csr: %s cost %.2f seconds ' % (bb_csr.shape, t3 - t2)
#B-*-B sim
#filename = dir_ + 'sim_res/path_count/%s.res' % stru_str[1:]#only need B-*-B
#data = np.genfromtxt(filename)
#row, col = [],[]
#
#row = [bid2ind[int(bid)] for bid in data[:,0]]
#col = [bid2ind[int(bid)] for bid in data[:,1]]
#bb_csr = csr((data[:,2], (row, col)), shape=[len(bids), len(bids)])

ubp_filename = dir_ + 'filter5_uid_pos_bid.txt'
data2 = np.loadtxt(ubp_filename, dtype=int)
rows = [uid2ind[int(r)] for r in data2[:,0]]
cols = [bid2ind[int(r)] for r in data2[:,1]]
ubp_csr = csr(([1] * len(rows), (rows, cols)), shape=[len(uids), len(bids)])
t4 = time.time()
print 'finish creating ubp_csr: %s ,cost %.2f seconds' % (ubp_csr.shape, t4 -t3)

def cal_sim(bid, cal_b_inds):
    sim = 0.0
    ind1 = bid2ind[bid]
    pn = 0
    for ind2 in cal_b_inds:
        if bb_csr[ind1,ind2] > 0.0:
            pn += 1.0
            sim += 2.0 * bb_csr[ind1,ind2] / (bb_csr[ind1,ind1] + bb_csr[ind2,ind2]) #path_sim
    return sim, sim/pn

#calculate similarity for every user, item pair
res, normalized_res = [], []
cal_start = start = time.time()
for cnt, uid in enumerate(uids):
    ub_inds = set(ubp_csr[uid2ind[uid],:].tocoo().col)#all non-zero indices for this uid
    can_bids = uid2can_bids.get(uid)
    if not can_bids:
        continue
    #print 'uid=%s, pos_bids=%s, can_bids=%s' % (uid, len(ub_inds), len(can_bids))
    for bc, bid in enumerate(can_bids):
        if (cnt * bc) % 1000000 == 0:
            end = time.time()
            print 'processing %s/%s user %s, %s/%s bid %s, cost %.2f seconds, total_cost %.2f minutes' % (cnt, len(uids), uid, bc, len(can_bids), bid, end - start, (end - cal_start) / 60.0)
            start = end
        sim, normalized_sim = cal_sim(bid, ub_inds)
        res.append((uid, bid, sim))
        normalized_res.append((uid, bid, normalized_sim))

def save(wfilename, res):
    fw = open(wfilename, 'w+')
    res = ['%s\t%s\t%s' % (u,i,s) for u,i,s in res]
    fw.write('\n'.join(res))
    fw.close()
    print 'finish saving %s pair of u-i(%s*%s) from %s, sim saved in %s' % (len(res), len(uids), len(bids), filename, wfilename)

print 'start saving sim res...'
t5 = time.time()
wfilename = dir_ + 'sim_res/path_sim/%s.res' % stru_str
save(wfilename, res)
wfilename = dir_ + 'sim_res/path_sim_norm/%s.res' % stru_str
save(wfilename, normalized_res)
t6 = time.time()
print 'finish saving two sim res and cal ends, cost %.2f seconds, total cost %.2f minutes' % (t6-t5, (t6-t1)/60.0)

