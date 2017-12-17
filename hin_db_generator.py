#coding=utf8
'''
    load raw data, create sqlite tables from raw data
'''
import sys
import sqlite3

from db_util import DB
from yelp_extractor import extract_from_json
from str_util import str2unicode

dir_ = 'data/yelp/'
database_filename = 'yelp2016_hin.db'
db = DB(dir_+database_filename)

CREATE_TABLE_SQLS = ['''CREATE TABLE if not exists entity(
                        id INTEGER PRIMARY KEY ASC autoincrement,
                        entity text NOT NULL, -- entity id, given from the raw data, usually in encripyted format
                        type_id INTEGER KEY NOT NULL, -- type id, represent the types of the entity
                        UNIQUE(entity,type_id) -- ensure the uniqueness of "entity type_id"
                        );
                     ''',
                     '''
                         CREATE TABLE if not exists type_id_map(
                         type_id INTEGER KEY NOT NULL,
                         type CHAR(20) DEFAULT ""
                         );
                     ''',
                     '''
                         CREATE TABLE if not exists rel_id_map(
                         rel_id INTEGER KEY NOT NULL,
                         relation CHAR(20) DEFAULT ""
                         );
                     ''',
                     '''
                         CREATE TABLE if not exists relation(
                         head_id INTEGER key NOT NULL,
                         tail_id INTEGER key NOT NULL,
                         rel_id INTEGER key NOT NULL,
                         UNIQUE(head_id, tail_id) ON CONFLICT IGNORE -- ensure the uniqueness of pair of entities
                         );
                     ''',
                     '''
                         CREATE TABLE if not exists type_rel_map(
                         head_tid INTEGER key NOT NULL, -- head type id
                         tail_tid INTEGER key NOT NULL, -- tail type id
                         rel_id INTEGER NOT NULL
                         );
                     ''',
                     #'''
                     #    CREATE TABLE if not exists review(
                     #    entity_id INTEGER primary key NOT NULL, -- the corresponding id in table entity
                     #    raw_text text DEFAULT "", -- raw text extracted from json
                     #    processed_text TEXT DEFAULT "" -- preprocessed text prepared for LDA
                     #    );
                     #''',
                    ]

INDEX_SQLS = [
            #'CREATE INDEX hr on relation (head_id, rel_id);',
            #'CREATE INDEX tr on relation (tail_id, rel_id);'
            'CREATE INDEX r on relation (rel_id);'
          ]


def create_table():
    conn = sqlite3.connect(dir_+database_filename)
    cur = conn.cursor()
    for sql in CREATE_TABLE_SQLS:
        cur.execute(sql)
        conn.commit()
    print 'creating table finished!'

def add_indexes():
    '''
        给数据库增加索引
    '''
    conn = sqlite3.connect(database_filename)
    cur = conn.cursor()
    for sql in INDEX_SQLS:
        cur.execute(sql)
        conn.commit()
        print 'add index, sql=%s' % sql
    print 'creating indexes finished!'

def get_type_map():
    '''
        get type map from database, list of pairs (type, type_id)
    '''
    sql = 'select type, type_id from type_id_map'
    types = db.get_list_by_sql(sql)
    return {r[0]:r[1] for r in types}

def generate_relations_from_types(types):
    '''
        generate relations by combining different type names
    '''
    relations = ['user_business_p', 'user_review', 'user_business_n', 'review_business_p', 'review_business_n', 'review_aspect_p', 'review_aspect_n']
    #relations = ['user_business_p', 'user_business_n', 'review_business_p', 'review_business_n', 'review_aspect_p', 'review_aspect_n']
    #for t1 in types:
    #    for t2 in types:
    #        if t1 == t2:
    #            continue
    #        relations.append('%s_%s' % (t1,t2))
    return relations

def load_all_entities():
    '''
        insert entities to table entity.
        insert id2type pair to type_id_table;
    '''
    names = ['business', 'user', 'review']
    entities = []
    ent2info = {}
    id2type = set()
    for ind, name in enumerate(names):
        ext_res = extract_from_json(ext_type=name)
        id2type.add((ind+1,name))
        for info in ext_res:
            entity_id = info['%s_id' % name]
            entities.append((entity_id, ind+1))
            com_key = "%s|%s" % (entity_id,ind+1)
            if com_key in ent2info:
                print 'duplicate id pair ' % com_key
                print 'saved_info...\n', ent2info[com_key]
                print 'duplicated info...\n', info
            else:
                ent2info[com_key] = info
        print 'finish loading %s %s entities' % (len(ext_res), name)
    print 'start to load %s entities' % len(entities)
    sql = '''
            INSERT INTO entity (entity, type_id) values (?, ?);
          '''
    db.insert_by_sql(sql, entities)
    id_types = sorted(list(id2type), key=lambda d:d[0], reverse=False)
    print 'start to load %s types' % (len(id_types))
    sql = 'INSERT INTO type_id_map (type_id, type) values (?, ?);'
    db.insert_by_sql(sql, id_types)

def get_entity_id_map(type_id):
    '''
        return a map of entity to ids given list of (entity, type_id) pairs
    '''
    sql = 'SELECT entity, id from entity where type_id=%s' % type_id
    ids = db.get_list_by_sql(sql)
    return {r[0]:r[1] for r in ids}

def load_review_aspects():
    '''
        load all the aspect review relations from the file generated by the aspect_extractor.py
    '''
    aspects_dir = dir_  + 'aspects/lda_v1/'
    topic_nums = [10,]
    rid2aspects = {}
    for tn in topic_nums:
        #review_id rate topic_ids raw text
        aspect_filename = 'review_topic%s.res' % tn
        type_ = 't%s_aspect' % tn
        lines = open(aspects_dir+aspect_filename,'r').readlines()
        parts = [l.strip().split() for l in lines if not l.startswith('#')]
        for info in parts:
            rid = str2unicode(info[0])
            topic_str = info[2]
            topic_ids = [('topic%s' % t.split(',')[0],type_) for t in topic_str.split('|')]
            rid2aspects.setdefault(rid, []).extend(topic_ids)
    return rid2aspects

def get_user_rate_polarity(uid2avg_rate, uid, rate):
    '''
        Given map uid2avg_rate, uid, return the polarity of current rate
        return Pos if rate >= uid2avg_rate, denoted as p, otherwise "n"
    '''
    return 'n' if rate < uid2avg_rate[uid] else 'p'

def load_user_average_rates():
    '''
        from user_rate stat file, load the user2average rates
    '''
    filename = 'user_avg_rate.txt'
    lines = open(dir_ + filename,'r').readlines()
    parts = [l.strip().split('\t') for l in lines]
    return {str2unicode(r[0]):float(r[1]) for r in parts}

def load_all_relations():
    '''
        load all relations from raw data and save into db as triplets
        save relation_id map
    '''
    type2id = get_type_map()
    relations = generate_relations_from_types(type2id.keys())
    print 'relations:', relations
    relations = zip(range(1,len(relations)+1), relations)
    sql = 'INSERT INTO rel_id_map (rel_id, relation) values (?, ?);'
    db.insert_by_sql(sql,relations)
    rel2id = {v:k for k,v in relations}

    u2eid = get_entity_id_map(type2id['user'])
    b2eid = get_entity_id_map(type2id['business'])
    r2eid = get_entity_id_map(type2id['review'])

    rev2aspects = load_review_aspects()
    uid2avg_rate = load_user_average_rates()

    topic_num = 10
    a2eid = get_entity_id_map(type2id['t%s_aspect' % topic_num])

    reviews = extract_from_json(ext_type='review')
    ub_trs, ur_trs, rb_trs, ra_trs = [], [], [], []
    #user_business and user_review relations
    for r in reviews:
        uid = r['user_id']
        bid = r['business_id']
        rid = r['review_id']
        rate = r['stars']
        aspects = rev2aspects[rid]
        polarity = get_user_rate_polarity(uid2avg_rate, uid,rate)

        ub_trs.append((u2eid[uid], b2eid[bid], rel2id['user_business_%s' % polarity]))
        #ur_trs.append((u2eid[uid], r2eid[rid], rel2id['user_review']))
        rb_trs.append((r2eid[rid], b2eid[bid], rel2id['review_business_%s' % polarity]))
        for asp, asp_type in aspects:
            if asp_type == 't%s_aspect' % topic_num:
                ra_trs.append((r2eid[rid], a2eid[asp], rel2id['review_aspect_%s' % polarity]))
    triplets = ub_trs + rb_trs + ra_trs
    sql = 'INSERT INTO relation (head_id, tail_id, rel_id) values (?, ?, ?);'
    print 'start to insert %s triplets' % len(triplets)
    db.batch_insert_by_sql(sql,triplets)

def load_aspects(filename):
    '''
        load aspects from file
        comment flag is "#"
        #review_id, rate, topic_str, raw_text
    '''
    lines = open(filename,'r').readlines()
    parts = [str2unicode(l.strip()).split() for l in lines if not l.startswith('#')]
    return parts

def save_aspects():
    '''
        load aspects related to every review, and save the topics as entity into db.
        generate aspect entity name by different topic model varing topic num
    '''

    aspects_dir = dir_ + 'aspects/lda_v1/'
    aspect_res = []
    id_types = set()
    for ind, topic_num in enumerate([5,10,20,30,40,50]):
        type_ = 't%s_aspect' % topic_num
        type_id = ind + 4
        aspect_res.extend([('topic%s' % n, type_id) for n in range(topic_num)])
        id_types.add((type_id,type_))
    print 'id_types:\n', id_types
    id_types = list(id_types)
    sql = 'insert into type_id_map (type_id, type) values (?, ?)'
    db.insert_by_sql(sql, id_types)
    print 'aspect_res:\n', aspect_res
    sql = 'insert into entity (entity, type_id) values (?, ?)'
    db.insert_by_sql(sql, aspect_res)

def save_categories():
    filename = dir_ + 'bc_type.txt'
    bc_rel = [l.strip().split() for l in open(filename,'r').readlines()]
    import pdb;pdb.set_trace()
    #bc_rel = [int(t) for t in r for r in bc_rel]
    #cat_tids = [(c,10) for c in cats]

    #print 'start to load %s types' % (len(id_types))
    #sql = 'INSERT INTO entity (entity, type_id) values (?, ?);'
    #db.insert_by_sql(sql,cat_tids)

    sql = 'INSERT INTO relation (head_id, tail_id, rel_id) values (?, ?, ?);'
    print 'start to insert %s triplets' % len(bc_rel)
    db.batch_insert_by_sql(sql,bc_rel)

def save_socials():
    '''
        save user socials into db
    '''
    filename = dir_ + 'user_social.txt'
    print 'save user socials, filename is ', filename
    lines = open(filename, 'r').readlines()
    socials = [l.strip().split() for l in lines]
    socials = [(int(u1), int(u2), 11) for u1,u2 in socials]
    sql = 'INSERT INTO relation (head_id, tail_id, rel_id) values (?, ?, ?);'
    print 'start to insert %s triplets' % len(socials)
    db.batch_insert_by_sql(sql, socials)


def main():
    if len(sys.argv) < 2:
        print 'please specify your loading type\n0: all \n1: create table only) \n2: add indexes\n3: load all entities \n4: load all relations\n5: test'
        sys.exit(0)
    else:
        type_ = int(sys.argv[1])
        if type_ == 1:
            create_table()
        elif type_ == 2:
            add_indexes()
        elif type_ == 3:
            load_all_entities()
        elif type_ == 4:
            load_all_relations()
        elif type_ == 5:
            test_select_many()
        elif type_ == 6:
            insert_to_db_from_file()
        elif type_ == 7:
            save_aspects()
        elif type_ == 10:
            save_categories()
        elif type_ == 11:
            save_socials()
        elif type_ == 0:
            create_table()
            add_indexes()
            load_all_entities()

def test_select_many():
    paras = ['mYSpR_SLPgUVymYOvTQd_Q','Sktj1eHQFuVa-M4bgnEh8g']
    sql = 'SELECT id from entity where entity=? and type_id=%s' % 1
    ids = get_entity_ids(paras,1)
    print ids

def insert_to_db_from_file():
    filename = 'data/kdd_sample/relation.txt'
    lines = open(filename, 'r').readlines()
    cols = [tuple(l.strip().split()) for l in lines]
    sql = 'INSERT INTO relation (head_id, tail_id, rel_id) values (?, ?, ?);'
    db.insert_by_sql(sql,cols)

if __name__ == '__main__':
    main()

