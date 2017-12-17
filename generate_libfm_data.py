#coding=utf8
'''
    generate data from mf-features for the libfm format
'''
import numpy as np
from utils import save_lines

def load_representation(t_dir, fnum):
    '''
        load user and item latent features generate by MF for every meta-graph
    '''
    ufilename = t_dir + 'uids.txt'
    bfilename = t_dir + 'bids.txt'
    uids = [int(l.strip()) for l in open(ufilename, 'r').readlines()]
    uid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in uids}

    bids = [int(l.strip()) for l in open(bfilename, 'r').readlines()]
    bid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in bids}

    ufiles, vfiles = [], []
    #if 'yelp-200k' in dir_:
    #    ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBStarsB_top1000_user.dat', 'UPBStateB_top1000_user.dat', 'UPBCityB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'UUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
    #    vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBStarsB_top1000_item.dat', 'UPBStateB_top1000_item.dat', 'UPBCityB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'UUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']
    #elif 'amazon-200k' in dir_:
    #    ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBBrandB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
    #    vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBBrandB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']

    ufiles.append('ratings_only_user.dat')
    vfiles.append('ratings_only_item.dat')

    for find, filename in enumerate(ufiles):
        ufs = np.loadtxt(t_dir + 'mf_features/path_count/' + filename, dtype=np.float64)
        cur = find * 10
        for uf in ufs:
            uid = int(uf[0])
            f = uf[1:]
            uid2reps[uid][cur:cur+10] = f

    for find, filename in enumerate(vfiles):
        bfs = np.loadtxt(t_dir + 'mf_features/path_count/' + filename, dtype=np.float64)
        cur = find * 10
        for bf in bfs:
            bid = int(bf[0])
            f = bf[1:]
            bid2reps[bid][cur:cur+10] = f
    return uid2reps, bid2reps

def generate_libfm_format(dir_, train_filename, test_filename):


    fnum = 120 if 'yelp' in dir_ else 90
    fnum = 10
    uid2features, bid2features = load_representation(dir_, fnum)

    generate(dir_ + train_filename, uid2features, bid2features, fnum)

    generate(dir_ + test_filename, uid2features, bid2features, fnum)

def generate(rating_filename, uid2features, bid2features, fnum):
    data = np.loadtxt(rating_filename)
    libfm = []
    for r in data:
        rate = r[2]
        uid, bid = int(r[0]),int(r[1])
        ufeatures = uid2features.get(uid, np.zeros(fnum))
        bfeatures = bid2features.get(bid, np.zeros(fnum))

        ufeatures = zip(range(fnum), ufeatures)
        bfeatures = zip(range(fnum,fnum*2), bfeatures)
        libfm.append('%s\t%s\t%s' % (rate, '\t'.join(['%s:%s' % (u,v) for u,v in ufeatures]), '\t'.join(['%s:%s' % (b,v) for b,v in bfeatures])))
    wfilename = '%s_%s.libfm' % (rating_filename, fnum)
    save_lines(wfilename, libfm)

for rnd in xrange(5):
    dir_ = 'data/yelp-50k/exp_split/%s/' % (rnd+1)
    ratings_filename = 'ratings'
    train_filename = '%s_train_%s.txt' % (ratings_filename, rnd+1)
    test_filename = '%s_test_%s.txt' % (ratings_filename, rnd+1)

    print 'process ', train_filename
    generate_libfm_format(dir_, train_filename, test_filename)
