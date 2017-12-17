#coding=utf8
'''
    split the uids into small blocks
'''
import numpy as np

dir_ = 'data/yelp/'
uid_filename = 'uids_filter5.txt'
ubp_filename = 'uid_pos_bid.txt'

bn = 32
step = 20000
uids = open(dir_ + uid_filename, 'r').readlines()
#ubp = np.loadtxt(dir_ + ubp_filename, dtype=np.int64)
#ubp = np.loadtxt(dir_ + ubp_filename, dtype=np.int64)
ubp = open(dir_ + ubp_filename, 'r').readlines()
ubp = [r.strip().split() for r in ubp]
ubp = [(int(u), int(b)) for u,b in ubp]
total = len(uids)
for i in xrange(0, total, step):
    ind = i / step + 1
    start = i
    end = start + step
    s_uids = uids[start:end]
    wfilename1 = dir_ + 'split_uids/uids_filter5_%s.txt' % ind
    fw = open(wfilename1, 'w+')
    fw.write(''.join(s_uids))
    fw.close()
    s_uids = set([int(r.strip()) for r in s_uids])
    #import pdb;pdb.set_trace()
    s_ubp = [r for r in ubp if r[0] in s_uids]
    wfilename2 = dir_ + 'split_uids/uid_pos_bid_filter5_%s.txt' % ind
    fw = open(wfilename2, 'w+')
    fw.write('\n'.join(['%s\t%s' % (u,b) for u, b in s_ubp]))
    fw.close()
    print 'finish save block %s in %s and %s ' % (ind, wfilename1, wfilename2)
