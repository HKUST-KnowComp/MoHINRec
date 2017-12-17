#coding=utf8
'''
    generate filter5 related ids
'''
import sys

#if dt == 'yelp-200k'
#    dir_ = 'data/%s/' % dt
#if dt == 'amazon-200k'
#    dir_ = 'data/amazon/'t


uid_filename = 'data/amazon-100k/uids.txt'
lines = open(uid_filename, 'r').readlines()
uids = set([r.strip() for r in lines])

bid_filename = 'data/amazon-100k/bids.txt'
lines = open(bid_filename, 'r').readlines()
bids = set([r.strip() for r in lines])

def filter_bids():
    ubp_filename = dir_ + 'uid_bid.txt'
    lines = open(ubp_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(r) for r in parts if r[0] in uids]
    wfilename = 'data/yelp-200k/uid_bid.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

def filter_comliments():
    uc_filename = dir_ + 'uid_comp_id.txt'
    lines = open(uc_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(r) for r in parts if r[0] in uids]
    wfilename = 'data/yelp-200k/uid_comp_id.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

def filter_social():
    social_filename = dir_ + 'user_social.txt'
    lines = open(social_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(r) for r in parts if r[0] in uids and r[1] in uids]
    wfilename = 'data/yelp-200k/user_social.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

def filter_rid_aspects():
    rid_filename = dir_ + 'filter5_pos_rids.txt'
    lines = open(rid_filename, 'r').readlines()
    rids = set([l.strip() for l in lines])
    filename = dir_ + 'rid_aid_weights.txt'
    lines = open(filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(r) for r in parts if r[0] in rids]
    wfilename = dir_ + 'filter5_rid_pos_aid_weights.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

def filter_rids():

    rid_filename = dir_ + 'filter5_neg_rids.txt'
    lines = open(rid_filename, 'r').readlines()
    rids = set([l.strip() for l in lines])

    filename = dir_ + 'filter5_uid_rid.txt'
    lines = open(filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(r) for r in parts if r[1] in rids]
    wfilename = dir_ + 'filter5_uid_neg_rid.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
#filter_social()

def filter_path_ratings():
    path = 'UBCatB'
    path_filename = dir_ + 'sim_res/path_count/%s.res' % path
    lines = open(path_filename, 'r').readlines()
    path_ub = set(['-'.join(l.strip().split()[:2]) for l in lines])

    ratings_filename = dir_ + 'ratings_filter5.txt'
    lines = open(ratings_filename, 'r').readlines()
    ratings = [l.strip().split() for l in lines]
    ratings = [r for r in ratings if '-'.join(r[:2]) in path_ub]

    wfilename = dir_ + 'ratings_%s' % path
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(['\t'.join(r) for r in ratings]))
    fw.close()

def filter_cat():
    bid_cat_filename = 'data/amazon/item_brand.txt'
    lines = open(bid_cat_filename,'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(r) for r in parts if r[0] in bids]
    wfilename = 'data/amazon-100k/bid_brand.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

#filter_social()
#filter_bids()
filter_cat()
#filter_comliments()
