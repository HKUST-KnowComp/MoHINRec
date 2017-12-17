#coding=utf
'''
    scripts to process yelp-200k
'''
import sys
from utils import save_lines

#dir_ = 'data/yelp-200k/'

#uid_filename = 'data/yelp-200k/uids.txt'
#lines = open(uid_filename, 'r').readlines()
#uids = set([r.strip() for r in lines])
#
#bid_filename = 'data/yelp-200k/bids.txt'
#lines = open(bid_filename, 'r').readlines()
#bids = set([r.strip() for r in lines])
#
#rating_filename = 'data/yelp-200k/ratings.txt'
#lines = open(rating_filename, 'r').readlines()
#ratings = [r.strip().split() for r in lines]

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
    filename = 'data/yelp/bid_%s.txt' % ot
    lines = open(filename,'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(p) for p in parts if p[0] in bids]
    wfilename = dir_ + 'bid_%s.txt' % ot
    save_lines(wfilename, res)

def generate_social():
    print 'run generate_social'
    filename = 'data/yelp/user_social.txt'
    lines = open(filename,'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(p) for p in parts if p[0] in uids and p[1] in uids]
    wfilename = dir_ + 'user_social.txt'
    save_lines(wfilename, res)

def generate_urb():
    '''
        uid-rid-bid-pos
        uid-rid-bid-neg
    '''
    print 'run generate_urb'
    upb_filename = dir_ + 'uid_pos_bid.txt'
    unb_filename = dir_ + 'uid_neg_bid.txt'
    lines = open(upb_filename, 'r').readlines()
    upb = set([l.strip() for l in lines])
    lines = open(unb_filename, 'r').readlines()
    unb = set([l.strip() for l in lines])

    ur_filename = 'data/yelp/filter5_uid_rid.txt'
    rb_filename = 'data/yelp/filter5_rid_bid.txt'
    lines = open(ur_filename,'r').readlines()
    parts = [l.strip().split() for l in lines]
    r2u = {r:u for u,r in parts}

    lines = open(rb_filename,'r').readlines()
    parts = [l.strip().split() for l in lines]

    urpb_res, urnb_res = [], []
    rids = []
    for r,b in parts:
        u = r2u[r]
        key = '%s\t%s' % (u,b)
        if key in upb:
            urpb_res.append('\t'.join((u,r,b)))
            rids.append(r)
        elif key in unb:
            urnb_res.append('\t'.join((u,r,b)))
            rids.append(r)

    urpb_filename = dir_ + 'uid_rid_pos_bid.txt'
    urnb_filename = dir_ + 'uid_rid_neg_bid.txt'

    save_lines(urpb_filename, urpb_res)
    save_lines(urnb_filename, urnb_res)

    rid_filename = dir_ + 'rids.txt'
    save_lines(rid_filename, rids)

def generate_ura():
    '''
        uid-rid-aid-pos
        uid-rid-aid-neg
    '''
    print 'run generate_ura...'
    urpb_filename = dir_ + 'uid_rid_pos_bid.txt'
    urnb_filename = dir_ + 'uid_rid_neg_bid.txt'

    lines = open(urpb_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    r2up = {r:u for u,r,b in parts}

    lines = open(urnb_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    r2un = {r:u for u,r,b in parts}


    ra_filename = 'data/yelp/filter5_rid_aid.txt'
    lines = open(ra_filename,'r').readlines()
    parts = [l.strip().split() for l in lines]

    urpa_res, urna_res = [], []
    aids = set()
    for r,a in parts:
        if r in r2up:
            urpa_res.append('\t'.join((r2up[r],r,a)))
            aids.add(a)
        elif r in r2un:
            urna_res.append('\t'.join((r2un[r],r,a)))
            aids.add(a)

    urpa_filename = dir_ + 'uid_rid_pos_aid.txt'
    urna_filename = dir_ + 'uid_rid_neg_aid.txt'

    save_lines(urpa_filename, urpa_res)
    save_lines(urna_filename, urna_res)

    aid_filename = dir_ + 'aids.txt'
    save_lines(aid_filename, aids)

def generate_uraw():
    '''
        uid-rid-aid-pos-weight
        uid-rid-aid-neg-weight
    '''
    print 'run generate_uraw...'
    urpb_filename = dir_ + 'uid_rid_pos_bid.txt'
    urnb_filename = dir_ + 'uid_rid_neg_bid.txt'

    lines = open(urpb_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    r2up = {r:u for u,r,b in parts}

    lines = open(urnb_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    r2un = {r:u for u,r,b in parts}


    ra_filename = 'data/yelp/filter5_rid_aid_weights.txt'
    lines = open(ra_filename,'r').readlines()
    parts = [l.strip().split() for l in lines]

    urpa_res, urna_res = [], []
    ind = 0
    for p in parts:
        try:
            r,a,w = p
            ind += 1
            if r in r2up:
                urpa_res.append('\t'.join((r2up[r],r,a,w)))
            elif r in r2un:
                urna_res.append('\t'.join((r2un[r],r,a,w)))
        except Exception as e:
            print parts[ind]
            print ind
            break

    urpa_filename = dir_ + 'uid_rid_pos_aid_weight.txt'
    urna_filename = dir_ + 'uid_rid_neg_aid_weight.txt'

    save_lines(urpa_filename, urpa_res)
    save_lines(urna_filename, urna_res)

def generate_all(dt):
    global dir_, uids, bids, ratings
    for ind in range(1,6):
        dir_ = 'data/%s/exp_split/%s/' % (dt, ind)
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

        for ot in ['cat', 'city', 'stars', 'state']:
            generate_bo(ot)

        get_user_average_rate(ratings)
        generate_polarity_pairs(ratings)

        generate_urb()

        generate_ura()

        generate_uraw()

        generate_social()

def generate_uid_comp():
    print 'run generate user compliment...'
    filename = 'data/cikm/yelp/uid_comp.txt'
    lines = open(filename,'r').readlines()
    parts = [l.strip().split() for l in lines]
    res = ['\t'.join(p) for p in parts if p[0] in uids]
    wfilename = dir_ + 'uid_comp.txt'
    save_lines(wfilename, res)

def generate_for_cikm():
    '''
        deal with cikm yelp dataset
    '''
    global dir_, uids, bids, ratings
    for ind in range(1,6):
        dir_ = 'data/cikm/yelp/exp_split/%s/' % ind
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

        #for ot in ['cat', 'city']:
        #    generate_bo(ot)

        #get_user_average_rate(ratings)
        #generate_polarity_pairs(ratings)

        #generate_social()

        generate_uid_comp()

if sys.argv[1] == 'avg':
    get_user_average_rate()
elif sys.argv[1] == 'polarity':
    generate_polarity_pairs()
#elif len(sys.argv) == 3:
#    ot = sys.argv[2]
#    generate_bo(ot)
elif sys.argv[1] == 'social':
    generate_social()
elif sys.argv[1] == 'ur':
    generate_ur()
elif sys.argv[1] == 'urb':
    generate_urb()
elif sys.argv[1] == 'ura':
    generate_ura()
elif sys.argv[1] == 'uraw':
    generate_uraw()
elif sys.argv[1] == 'all':
    dt = sys.argv[2]
    generate_all(dt)
elif sys.argv[1] == 'cikm':
    generate_for_cikm()
