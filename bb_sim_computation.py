#coding=utf8
'''
    Given commuting matrix for B-*-B meta paths, calcuate the similarity for later utilization
'''
import time
from scipy.sparse import csr_matrix as csr

from utils import reverse_map, generate_adj_mat

stru_str = 'BCityB'
dir_ = 'data/yelp/samples/'
t1 =  time.time()
filename = 'commuting_mat/%s.res' % stru_str
lines = open(dir_+filename, 'r').readlines()
t2 =  time.time()
print 'load %s records from file cost %2.f seconds' % (len(lines), t2 - t1)
parts = [l.strip().split() for l in lines]
bbs = [(int(b1), int(b2), int(num)) for b1, b2, num in parts]
bids = set([b1 for b1, _,_ in bbs])
bid2ind = {v:k for k,v in enumerate(bids)}
ind2bid = reverse_map(bid2ind)
bb,bb_t = generate_adj_mat(bbs, bid2ind, bid2ind, is_weight=True)
t3 =  time.time()
print 'preprocessing cost %2.f seconds' % (t3 - t2)

sim_res = []
cn = 0
t1 = time.time()
for b1 in bids:
    ind1 = bid2ind[b1]
    cnt1 = bb[ind1,ind1]
    cols = bb[ind1,:].tocoo().col
    for ind2 in cols:
        cn += 1
        b2 = ind2bid[ind2]
        if cn % 1000000 == 0:
            t2 = time.time()
            print 'processing pair: (%s-%s), cn=%s/%s, cost %.2f seconds' % (b1, b2, cn, len(bbs), t2 - t1)
            t1 = t2
        sim = 2.0 * bb[ind1, ind2] / (cnt1 + bb[ind2,ind2])
        sim_res.append('%s\t%s\t%s' % (b1,b2,sim))

t3 = time.time()
wfilename = 'sim_res/%s.res' % stru_str
fw = open(dir_ + wfilename, 'w+')
fw.write('\n'.join(sim_res))
fw.close()
t4 = time.time()
print 'save to %s, cost %2.f seconds' % (dir_+wfilename, t4 - t3)
