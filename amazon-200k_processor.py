#coding=utf
'''
    scripts to process yelp-200k
'''
import sys
from utils import save_lines

def get_user_average_rate(ratings):
    print 'run get_user_average_rate...'
    uid2rates = {}
    for u, b, r in ratings:
        uid2rates.setdefault(u, []).append(float(r))
    uid2avg = [(uid, sum(rates) * 1.0 / len(rates)) for uid, rates in uid2rates.items()]
    filename = 'user_avg_rate.txt'
    fw = open(dir_+filename, 'w+')
    fw.write('\n'.join(['%s\t%s\t%s' % (uid,round(avg,2), len(uid2rates[uid])) for uid,avg in uid2avg]))

def generate_polarity_pairs(ratings):
    '''
        uid_pos_bid.txt
        uid_neg_bid.txt
    '''
    print 'run generate_polarity_pairs...'
    filename = 'user_avg_rate.txt'
    lines = open(dir_ + filename,'r').readlines()
    parts = [l.strip().split() for l in lines]
    uid2avg_rate = {k:float(v) for k, v, _ in parts}
    upb, unb = [], []
    for u, b, r in ratings:
        if float(r) < uid2avg_rate[u]:
            unb.append('%s\t%s' % (u,b))
        else:
            upb.append('%s\t%s' % (u,b))

    upb_filename, unb_filename = 'uid_pos_bid.txt', 'uid_neg_bid.txt'

    save_lines(dir_ + upb_filename, upb)
    save_lines(dir_ + unb_filename, unb)

def generate_bo(ot):
    '''
        uid_cat, uid_state, uid_stars, uid_city
    '''
    print 'run generate_', ot
    filename = 'data/amazon-200k/bid_%s.txt' % ot
    lines = open(filename,'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(p) for p in parts if p[0] in bids]
    wfilename = dir_ + 'bid_%s.txt' % ot
    save_lines(wfilename, res)

def generate_polarity_urba():
    '''
        uid-rid-bid-pos
        uid-rid-bid-neg
    '''
    print 'run generate_polarity_urba'

    upb_filename = dir_ + 'uid_pos_bid.txt'
    unb_filename = dir_ + 'uid_neg_bid.txt'
    lines = open(upb_filename, 'r').readlines()
    upb = set([l.strip() for l in lines])
    lines = open(unb_filename, 'r').readlines()
    unb = set([l.strip() for l in lines])

    ubra_filename = 'data/amazon/ubra.txt'
    lines = open(ubra_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]

    urpb_res, urnb_res = [], []
    urpa_res, urna_res = [], []
    urpaw_res, urnaw_res = [], []
    rids,aids = [],set()
    for u,b,r,rate, astr in parts:
        key = '%s\t%s' % (u,b)
        ts = astr.strip().split('|')
        ts = [t.split(',') for t in ts]
        r = r.replace('review_', '')
        if key in upb:
            urpb_res.append('\t'.join((u,r,b)))
            for t,w in ts:
                urpa_res.append('\t'.join((u,r,t)))
                urpaw_res.append('\t'.join((u,r,t,w)))
                aids.add(t)
            rids.append(r)
        elif key in unb:
            urnb_res.append('\t'.join((u,r,b)))
            for t,w in ts:
                urna_res.append('\t'.join((u,r,t)))
                urnaw_res.append('\t'.join((u,r,t,w)))
                aids.add(t)
            rids.append(r)

    urpb_filename = dir_ + 'uid_rid_pos_bid.txt'
    urnb_filename = dir_ + 'uid_rid_neg_bid.txt'
    save_lines(urpb_filename, urpb_res)
    save_lines(urnb_filename, urnb_res)


    urpa_filename = dir_ + 'uid_rid_pos_aid.txt'
    urna_filename = dir_ + 'uid_rid_neg_aid.txt'
    save_lines(urpa_filename, urpa_res)
    save_lines(urna_filename, urna_res)


    urpaw_filename = dir_ + 'uid_rid_pos_aid_weight.txt'
    urnaw_filename = dir_ + 'uid_rid_neg_aid_weight.txt'
    save_lines(urpaw_filename, urpaw_res)
    save_lines(urnaw_filename, urnaw_res)

    rid_filename = dir_ + 'rids.txt'
    aid_filename = dir_ + 'aids.txt'
    save_lines(rid_filename, rids)
    save_lines(aid_filename, aids)

def generate_all():
    global dir_, uids, bids, ratings
    for ind in range(1,6):
        dir_ = 'data/amazon-200k/exp_split/%s/' % ind
        print 'process %s ...' % dir_
        rating_filename = dir_ + 'ratings_train_%s.txt' % ind
        lines = open(rating_filename, 'r').readlines()
        ratings = [l.strip().split() for l in lines]
        uids = set([r[0] for r in ratings])
        bids = set([r[1] for r in ratings])
        uwfilename = dir_ + 'uids.txt'
        bwfilename = dir_ + 'bids.txt'
        save_lines(uwfilename, uids)
        save_lines(bwfilename, bids)

        for ot in ['cat', 'brand']:
            generate_bo(ot)

        get_user_average_rate(ratings)
        generate_polarity_pairs(ratings)

        generate_polarity_urba()

if sys.argv[1] == 'avg':
    get_user_average_rate()
elif sys.argv[1] == 'polarity':
    generate_polarity_pairs()
elif len(sys.argv) == 3:
    ot = sys.argv[2]
    generate_bo(ot)
elif sys.argv[1] == 'ur':
    generate_ur()
elif sys.argv[1] == 'urb':
    generate_urb()
elif sys.argv[1] == 'ura':
    generate_ura()
elif sys.argv[1] == 'uraw':
    generate_uraw()
elif sys.argv[1] == 'all':
    generate_all()
