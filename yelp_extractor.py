#coding=utf8
'''
    util that extract the specific data of yelp
'''
import json

def get_filename(ext_type):
    dir_ = 'data/yelp/2016/'
    if ext_type == 'review':
        return dir_ + 'yelp_academic_dataset_review.json'
    if ext_type == 'business':
        return dir_ + 'yelp_academic_dataset_business.json'
    if ext_type == 'user':
        return dir_ + 'yelp_academic_dataset_user.json'

def extract_from_json(ext_type='review', is_simapling=False):
    file_path = get_filename(ext_type)

    lines = open(file_path).readlines()
    res = []
    for l in lines:
        item_info = json.loads(l.strip())
        res.append(item_info)
        if is_simapling and len(res) > 100:
            break
    print 'extract %s, total=%s' % (ext_type, len(res))
    return res

def extract_user_compliment():
    users = extract_from_json(ext_type='user')
    uid_map_filename = 'data/yelp/user_id_map.txt'
    lines = open(uid_map_filename, 'r').readlines()
    parts = [l.strip().split(',') for l in lines]
    user2id = {r[1]:int(r[0]) for r in parts}

    com_map_filename = 'data/yelp/compliment_id_map.txt'
    lines = open(com_map_filename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    comp2id = {r[1]:int(r[0]) for r in parts}
    res = []
    for user in users:
        uid = user['user_id']
        compliments = user['compliments']
        #import pdb;pdb.set_trace()
        for k, v in compliments.items():
            res.append('%s\t%s\t%s' % (user2id[uid], comp2id[k], 1))

    wfilename = 'data/yelp/uid_comp_id.txt'
    fw = open(wfilename, 'w+')
    fw.write('\n'.join(res))
    fw.close()

#extract_user_compliment()

