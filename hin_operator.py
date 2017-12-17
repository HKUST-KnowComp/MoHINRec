#coding=utf8
'''
    operators for hin, based on sqlite3
'''
import time
import numpy as np

from db_util import DB
import pandas as pd
#database_filename = 'data/kdd_sample/kdd_sample.db'
database_filename = 'data/yelp/yelp2016_hin.db'

class HIN(object):
    '''
        Build and preserve HIN
        Provide all interfaces to access entities or relations in HIN.
    '''
    def __init__(self, db_filename='', layer_relations=[]):
        self.db = DB(database_filename)
        self.load_dfs(layer_relations)

    def load_dfs(self, layer_relations):
        '''
            load needed tables into pandas dataframes from the sqlite3 db
        '''
        start = time.time()
        sql = 'select * from entity'
        self.ent_df = pd.read_sql_query(sql, self.db.conn)
        sql = 'select * from relation'
        self.rel_df = pd.read_sql_query(sql, self.db.conn)
        sql = 'select * from type_id_map'
        self.tim_df = pd.read_sql_query(sql, self.db.conn)
        sql = 'select * from rel_id_map'
        self.rim_df = pd.read_sql_query(sql, self.db.conn)
        sql = 'select * from type_rel_map'
        self.trm_df = pd.read_sql_query(sql, self.db.conn)
        self.rel_dfs = []
        #for rels in layer_relations:
        #    rel_ids = [self.rim_df[self.rim_df['relation'] == r].values[0,0] for r in rels]
        #    self.rel_dfs.append(rel_df[rel_df['rel_id'].isin(rel_ids)])


        end = time.time()
        print 'finish loading all data from database, cost %.2fmin' % ((end-start) / 60.0)

    def get_children_instances(self, p_obj_id, child_tid, rel_type=''):
        '''
            given child_type, instance of the parent object,
            return all the nodes of child_type, whose parent is current instance, i.e. the p_obj
            currently, we need to do at most twice the times of db quering, can be improved
            how about by sql query or relation
            the operation is based on one assumption: for any two types or two instances, there is only one relation, or the code is incorrect
            If the rel_type is given, then we use the rel_type directly.
        '''
        rel_id = ''
        if rel_type:
            rel_id = self.get_rel_id_by_name(rel_type)
        else:
            p_tid = self.get_ins_type_id(p_obj_id)
            rel_id = self.get_rel_by_types(p_tid, child_tid)

        children_ids = self.get_connected_ids(p_obj_id, rel_id)

        return set(children_ids) if children_ids else set()

    def get_children_instances_df(self, p_obj_id, child_tid, rel_type='', layer_id=1):
        '''
        '''
        rel_id = ''
        if rel_type:
            rel_id = self.rim_df[self.rim_df['relation'] == rel_type].values[0][0]
        else:
            p_tid = self.get_ins_type_id(p_obj_id)
            rel_id = self.get_rel_by_types(p_tid, child_tid)

        start_time = time.time()

        rel_df = self.rel_dfs[layer_id-1]
        ids = rel_df[((rel_df['tail_id'] == p_obj_id) | (rel_df['head_id'] == p_obj_id)) & (rel_df['rel_id'] == rel_id)].values
        ids = ids[:,(0,1)].flatten()

        cost = time.time() - start_time
        children_ids = [r for r in ids if r != p_obj_id]

        return (set(children_ids), cost) if children_ids else (set(), cost)

    def get_rel_id_by_name(self,name):
        '''
            given relation name, return rel_id
        '''
        sql = 'select rel_id from rel_id_map where relation = \"%s\"' % name
        res = self.db.get_list_by_sql(sql)
        return res[0][0]

    def get_connected_ids(self, oid, rel_id):
        '''
            Given one obj id and relation id, return the connected obj ids, no matter which the direction is
        '''
        sql = 'select head_id, tail_id from relation where (head_id=%s or tail_id=%s) and rel_id=%s' % (oid, oid, rel_id)
        res = self.db.get_list_by_sql(sql)
        if not res:
            return []
        ret = []
        for r in res:
            tmp = list(r)
            tmp.remove(oid)
            ret.append(tmp.pop())#remove oid, the remaining is the connected one
        return ret

    def get_triplets_by_rel_id(self, rel_id):
        '''
            return head_id, tail_id, rel_id, given the rel_id
        '''
        return self.rel_df[self.rel_df['rel_id'] == rel_id].values

    def get_type_id_by_name(self, name):
        '''
            given type name, return the type_id in the table type_id_map
        '''
        #sql = 'select type_id from type_id_map where type=\"%s\"' % name
        #res = self.db.get_list_by_sql(sql)
        #return res[0][0] if res else None
        return self.tim_df[self.tim_df['type'] == name].values[0,0]

    def get_ins_type_id(self,ins_id):
        '''
            return instance type_id, given instance_id
        '''
        sql = 'select type_id from entity where id=%s' % ins_id
        res = self.db.get_list_by_sql(sql)
        return res[0][0] if res else None

    def get_rel_by_types(self, tid1, tid2):
        '''
           return relation id, gven ids of two types
        '''
        sql = 'select rel_id from type_rel_map where (head_tid=%s and tail_tid=%s) or (tail_tid=%s and head_tid=%s)' % (tid1, tid2, tid1, tid2)
        res = self.db.get_list_by_sql(sql)
        return res[0][0] if res else None

    def get_tail_ids(self, head_id, rel_id):
        '''
            return all the tail ids, given head_id and relation id
        '''
        sql = 'select tail_id from relation where head_id=%s and rel_id=%s' % (head_id, rel_id)
        res = self.db.get_list_by_sql(sql)
        return [r[0] for r in res] if res else []

    def get_tail_ids_by_head_ids(self, hids, rel_id):
        tids = self.rel_df[(self.rel_df['head_id'].isin(hids)) &(self.rel_df['rel_id'] == rel_id)].values[:,1]
        return tids

    def get_relations_by_head_ids(self, hids, rel_id):
        relations = self.rel_df[(self.rel_df['head_id'].isin(hids)) &(self.rel_df['rel_id'] == rel_id)].values
        return relations

    def get_head_ids(self, tail_id, rel_id):
        '''
            return all the head ids, given tail id and relation id
        '''
        sql = 'select head_id from relation where tail_id=%s and rel_id=%s' % (tail_id, rel_id)
        #print sql
        res = self.db.get_list_by_sql(sql)
        return [r[0] for r in res] if res else []

    def get_entity_id_df(self, name='', type_id=''):
        res = self.ent_df[(self.ent_df['entity'] == name) & (self.ent_df['type_id'] == type_id)].values
        return res[0][0]

    def get_entity_id(self, name='', type_id=''):
        '''
            return the global id when giving name and type_id
        '''
        sql = 'select id from entity where entity=\"%s\" and type_id=%s' % (name, type_id)
        res = self.db.get_list_by_sql(sql)
        return res[0][0] if res else None

    def get_entity_ids_by_type(self, name='', limit=0, rand_sample=True):
        type_id = self.get_type_id_by_name(name)
        eids = self.ent_df[self.ent_df['type_id'] == type_id].values[:,0]
        return np.random.choice(eids, size=limit, replace=False) if limit else eids

    def get_entitities_by_type(self, name=''):
        type_id = self.get_type_id_by_name(name)
        entities = self.ent_df[self.ent_df['type_id'] == type_id].values
        return entities


if __name__ == '__main__':
    hin = HIN()
    #print hin.get_children_instances(16, 3) #should return 10
    #print hin.get_children_instances(16, 2) #should return 7
    #print hin.get_children_instances(2, 4) #should return 12,13
    #print hin.get_children_instances_df(2, 4) #should return 12,13
    #print hin.get_children_instances(1, 4) #should return 12,13
    #print hin.get_children_instances_df(1, 4) #should return 12,13
    #print hin.get_children_instances(8, 4) #should return 12


