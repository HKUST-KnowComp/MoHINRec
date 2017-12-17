#coding=utf8
'''
    scripts used to generate data for sample users
'''
dir_ = 'data/yelp/'
ufilename = 'uids.txt'
pos_bid_filename = 'pos_bids.txt' # bids of user_business_p

bids = set([l.strip() for l in open(dir_+pos_bid_filename, 'r').readlines()])

#same processing for cat, city, stars, review_count
t = 'cat'
cat_filename = 'data/yelp/bid_%s.txt' % t
bid_cat = [l.strip().split() for l in open(cat_filename, 'r').readlines()]
bid_cat = [(b,c) for b,c in bid_cat if b in bids]
res = ['\t'.join(r) for r in bid_cat]
wfilename = 'data/yelp/pos_bid_%s.txt' % t
fw = open(wfilename, 'w+')
fw.write('\n'.join(res))
fw.close()
