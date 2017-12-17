#coding=utf8
'''
    processing methods for ranking-based exp
'''

from utils import save_lines

dir_ = 'data/yelp-50k/exp_split/1/'
train_filename = 'ratings_train_1.txt'

def pair2ordered(pair_list, uid):
    n = len(pair_list)
    res = []
    for i in range(n):
        vi, r1 = pair_list[i].split()
        r1 = float(r1)
        for j in range(i+1, n):
            vj, r2 = pair_list[j].split()
            r2 = float(r2)
            if int(r1) > int(r2):
                res.append((uid, vi, vj, str(int(r1) - int(r2))))
            elif int(r1) < int(r2):
                res.append((uid, vj, vi, str(int(r2) - int(r1))))
    return res

def generate_train_data():
    lines = open(dir_ + train_filename).readlines()
    uid2pairs = {}
    res = []
    for l in lines:
        parts = l.strip().split()
        uid = parts[0]
        pair = '\t'.join(parts[1:])
        uid2pairs.setdefault(uid, []).append(pair)
    for uid, pair in uid2pairs.items():
        res.extend(pair2ordered(pair, uid))
    wfilename = train_filename.replace('.txt', '_ranking.txt')
    res = ['\t'.join(r) for r in res]
    save_lines(dir_+wfilename, res)

if __name__ == '__main__':
    generate_train_data()
