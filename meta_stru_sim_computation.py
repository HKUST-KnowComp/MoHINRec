#coding=utf8
'''
    computing similarity based on meta_structures
'''
import itertools
import operator
import time

from str_util import str2unicode

from hin_operator import HIN

#class Node(object):
#    '''
#        when instance = -1, representing the Node Type itself, rather than any instance.
#    '''
#    def __init__(self, T='', instance=-1):
#        self.T = T
#        self.instance = instance

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
        self.layers = structure_layers
        self.relations = relations
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
    global total_db_cost, child_id_cost
    yelp_dir = 'data/yelp/sim_users/'
    #test_meta_structure_str = 'user-review-business,aspect-review-user'
    test_meta_structure_str = 'user-review-business,t10_aspect-review-user'
    test_relation_str = 'user_review|review_business_p,review_aspect_p|review_business_p,review_aspect_p|user_review'
    #test_relation_str = 'user_review|review_business,review_aspect|review_business,review_aspect|user_review'
    test_users = open('test_users2.txt', 'r').readlines()
    test_users = [str2unicode(l.strip()) for l in test_users]
    type_id = 2
    relations = [tuple(r.split(',')) for r in test_relation_str.split('|')]
    hin = HIN(layer_relations=relations)
    s_tree = ETree(structure_str=test_meta_structure_str, isType=True, relation_str=test_relation_str)

    print 's_tree.layers=', s_tree.layers
    print 's_tree.relations=', s_tree.relations
    for user_name in test_users:
        start = time.time()
        total_db_cost = 0.0
        child_id_cost = 0.0
        uid = hin.get_entity_id_df(user_name, type_id)
        if uid in [86008, 86005]:
            continue
        fw = open(yelp_dir+'user_%s.res' % uid, 'w+')
        gsr_tree = ETree()
        gsr_tree.add_layer((uid,))
        print 'start to process user %s(uid=%s)' % (user_name, uid)
        res = traverse_trees(s_tree, gsr_tree, 1.0, 1)
        weights, cnts = {}, {}
        for g, w in res:
            if g:
                pair = '%s-%s' % (uid, g.layers[g.ds-1][0])
                weights[pair] = weights.get(pair, 0.0) + w
                cnts[pair] = cnts.get(pair, 0) + 1
                fw.write('instance: g=%s, w=%s\n' % (g.layers, w))

        time_str = 'finish cal all similarities for user %s(uid=%s), cost %.2fmin(db_cost=%.2fmin, child_id_cost=%.2fmin)\n' % (user_name, uid, (time.time() - start) / 60.0, total_db_cost/60.0, child_id_cost/60.0)
        print time_str
        fw.write(time_str)
        fw.write('*************the weights of pair are in the following(pair_id, number of instances, weight)**********\n')
        weights = sorted(weights.items(), key=lambda d:d[1], reverse=True)
        for k, v in weights:
            fw.write('%s:%s,%s\n' % (k, cnts[k], v))
        fw.close()

if __name__ == '__main__':
    main()
