#coding=utf8
'''
    based on the second optimization scheme in KDD16 paper, implement the i-table processing
    the process is very similar to the similarity calculation process, which also needs to traverse the ETree
    in fact, we are doing the traversal given partial meta-structure
'''
import time
import itertools
import operator

import pandas as pd

from hin_operator import HIN
global hin, total_db_cost, child_id_cost

class ETree(object):
    '''
        preserve
        layers: perseve the nodes in every layer of one meta_structures.
                The nodes are instances when it is used to preserve one instance of meta_strucuture
                Element is a tuple, saving the corresponding nodes or instances in every layer.
    '''
    def __init__(self, structure_layers=[], is_structure=False, relations=[]):
        '''
            isType=True, load the meta_structure
            isType=False, instance of meta_structure
        '''
        self.layers = structure_layers[:]#value passing
        self.relations = relations[:]
        self.ds = len(structure_layers)
        self.is_structure = is_structure

    def _load_structures(self, structure_str, relation_str):
        l_types = structure_str.split('-')
        l_types = [tuple(l.split(',')) for l in l_types]
        layers = []
        for ts in l_types:
            layers.append(tuple([hin.get_type_id_by_name(t) for t in ts]))
        self.layers = layers
        self.ds = len(self.layers)
        relations = relation_str.split('|')
        print relations
        self.relations = [tuple(r.split(',')) for r in relations]
        print self.relations

    def add_layer(self, layer):
        self.layers.append(layer)
        self.ds += 1

def traverse_trees(s_tree, g_tree, w, layer_id):
    if layer_id == s_tree.ds:
        return [(g_tree, w)]

    global total_db_cost, child_id_cost
    instances = {}
    ind = 0
    #ind equals to 0 or 1.
    for type_id in s_tree.layers[layer_id]:
        nt_instances = []
        for obj_id in g_tree.layers[layer_id - 1]:
            #every time at most two objs because of the design of meta structures
            rel_type = s_tree.relations[layer_id - 1][ind]
            ind += 1
            #print 'layer_id=%s,current_id=%s,rel_type=%s, ind=%s,type_id=%s' % (layer_id,obj_id,rel_type,ind,type_id)
            db_start = time.time()
            ch_ins_ids, cost = hin.get_children_instances_df(obj_id, type_id, rel_type, layer_id)
            nt_instances.append(ch_ins_ids)
            total_db_cost += (time.time() - db_start)
            child_id_cost += cost

        instances[type_id] = list(set.intersection(*nt_instances))
        if not instances[type_id]:
            print 'search ends..., layer_id=%s, type_id=%s, g_layers=%s' % (layer_id, type_id, g_tree.layers)
            return [(None, -1)]
    new_layers = list(itertools.product(*instances.values()))
    w_ = w * 1.0 / len(new_layers)
    rtn = []
    for layer in new_layers:
        g_ = ETree()
        g_.layers = list(g_tree.layers)
        g_.ds = g_tree.ds
        g_.add_layer(layer)
        res = traverse_trees(s_tree, g_, w_, layer_id + 1)
        rtn.extend(res)
    return rtn

def main():
    dir_ = 'data/yelp/'
    meta_structure_str = 'business,t10_aspect-review-user-business'
    relation_str = 'review_business_p,review_aspect_p|user_review|user_business_p'
    global hin, total_db_cost, child_id_cost

    relations = [r.split(',') for r in relation_str.split('|')]
    hin = HIN(layer_relations=relations)

    l_types = [l.split(',') for l in meta_structure_str.split('-')]
    layers = []
    for ts in l_types:
        layers.append(tuple([hin.get_type_id_by_name(t) for t in ts]))

    s_tree = ETree(structure_layers=layers, is_structure=True, relations=relations)
    print 's_tree.layers=', s_tree.layers

    biz_ids = hin.get_entities_by_type('business', limit=100000)
    aspect_ids = hin.get_entities_by_type('t10_aspect', limit=10)
    pairs = list(itertools.product(*[biz_ids,aspect_ids]))
    tables = []
    print '%s b-a pairs' % (len(pairs))
    cnt = 0
    for bid, aid in pairs:
        #import pdb;pdb.set_trace()
        cnt += 1
        total_db_cost = 0.0
        child_id_cost = 0.0
        gsr_tree = ETree()
        gsr_tree.add_layer((bid, aid))
        print 'cnt=%s,start to process pair(biz-aspect): %s-%s)' % (cnt,bid, aid)
        start = time.time()
        res = traverse_trees(s_tree, gsr_tree, 1.0, 1)
        end = time.time()
        print 'finish pair: %s-%s, %s targets, cost %.2fmin' % (bid,aid, len(res),(end-start)/60.0)
        weights = {}
        for g, w in res:
            if g:
                tbid = g.layers[-1][0]
                weights[tbid] = weights.get(tbid,0.0) + w
        for tbid, w in weights.items():
            tables.append((bid,aid,tbid,w))
    df = pd.DataFrame(tables,columns=['business_eid','aspect_eid','business_eid','weight'])
    df.to_csv(dir_ + '3table_all.csv')

if __name__ == '__main__':
    main()
