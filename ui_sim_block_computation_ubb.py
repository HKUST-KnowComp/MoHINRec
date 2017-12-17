#coding=utf8
'''
    calculate the user-item similarity based on different meta-structures
    this script is for the U-pos-B-*-B style meta-path by block computation
'''


import sys
import time
import numpy as np
from scipy.sparse import csr_matrix as csr
from scipy.sparse import csc_matrix as csc

from utils import reverse_map


stru_str = 'BCatB'
if len(sys.argv) == 2:
    stru_str = sys.argv[1]
    print 'stru_str is specified: ', stru_str
else:
    print 'please speficy the stru_str'
    sys.exit(0)

dir_ = 'data/yelp/'

filename = dir_ + 'uids.txt'
uids = [int(l.strip()) for l in open(filename, 'r').readlines()]
uids = set(uids)
uid2ind = {v:k for k,v in enumerate(uids)}
ind2uid = reverse_map(uid2ind)

filename = dir_ + 'pos_bids.txt'
lines = open(filename, 'r').readlines()
bids = [int(l.strip()) for l in lines]
bids = list(set(bids))
bid2ind = {v:k for k,v in enumerate(bids)}

#B-*-B sim
filename = dir_ + 'commuting_mat/%s.res' % stru_str
data = np.genfromtxt(filename)
row, col = [],[]
row = [bid2ind[int(bid)] for bid in data[:,0]]
col = [bid2ind[int(bid)] for bid in data[:,1]]
bb_csr = csr((data[:,2], (row, col)), shape=[len(bids), len(bids)])
print 'bb_csr: ', bb_csr.shape
del data

ubp_filename = dir_ + 'uid_pos_bid.txt'
data2 = np.loadtxt(ubp_filename, dtype=int)
rows = [uid2ind[int(r)] for r in data2[:,0]]
cols = [bid2ind[int(r)] for r in data2[:,1]]
ubp_csr = csr(([1] * len(rows), (rows, cols)), shape=[len(uids), len(bids)])
print 'ubp_csr: ', ubp_csr.shape
del data2

def cal_sim(bid, cal_b_inds):
    sim = 0.0
    ind1 = bid2ind[bid]
    for ind2 in cal_b_inds:
        sim += bb_csr[ind1,ind2] #sim has been calculated
    return sim

cnt = 0
#calculate similarity for every user, item pair
res = []
start = time.time()
for uid in uids[:80000]:
    ub_inds = set(ubp_csr[uid2ind[uid],:].tocoo().col)#all non-zero indices for this uid
    for bid in bids:
        cnt += 1
        if cnt % 1000000 == 0:
            end = time.time()
            print 'processing %s/%s, cost %.2f seconds' % (cnt, len(uids) * len(bids), end - start)
            start = end
        nb_inds = set(bb_csr[bid2ind[bid],:].tocoo().col)#get the non-zero uid
        cal_b_inds = nb_inds.intersection(ub_inds)# the shared neighbours between uid and bid
        if not bool(cal_b_inds):
            continue
        sim = cal_sim(bid, cal_b_inds)
        res.append((uid, bid, sim))
wfilename = dir_ + 'sim_res/U%s.res' % stru_str
fw = open(wfilename, 'w+')
res = ['%s\t%s\t%s' % (u,i,s) for u,i,s in res]
fw.write('\n'.join(res))
fw.close()
print '%s pair of u-i(%s*%s) from %s, sim saved in %s' % (len(res), len(uids), len(bids), filename, wfilename)
