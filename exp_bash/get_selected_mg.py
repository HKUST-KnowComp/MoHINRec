#!/project/edbg/data/anaconda2/bin/python
#coding=utf8
'''
    Given the res filename of W and P, get the selected and removed meta-graphs for users and items, respectively
'''
import numpy as np
import sys

def pretty_order(a):
    a = sorted(list(a), key=lambda d:int(d[1]))
    return ['_'.join(r) for r in a]

def run(filename):
    if 'yelp' in filename:
        ind2mg = {1: 'M1', 2: 'M9', 3: 'M9', 4: 'M4', 5: 'M7', 6: 'M6', 7: 'M5', 8: 'M3', 9: 'M3', 10: 'M2', 11: 'M8', 12: 'M8'}
    elif 'amazon' in filename:
        ind2mg = {1: 'M1', 2: 'M6', 3: 'M6', 4: 'M3', 5: 'M4', 6: 'M2', 7: 'M2', 8: 'M5', 9: 'M5'}

    data = np.loadtxt(filename)
    m = data.shape[0]
    data = data.reshape(m, -1)
    a = data[:,0]
    zero_inds = np.where(a == 0)[0]
    nonzero_inds = np.nonzero(a)[0]
    useless_inds = zero_inds / 10 + 1
    useful_inds = nonzero_inds / 10 + 1
    N = m / 20
    useful_u = set([ind2mg[r] for r in useful_inds if r <= N])
    useless_u = set([ind2mg[r] for r in useless_inds if r <= N])
    useful_i = set([ind2mg[r- N] for r in useful_inds if r > N])
    useless_i = set([ind2mg[r - N] for r in useless_inds if r > N])
    useless_u = [r for r in useless_u if r not in useful_u]
    useless_i = [r for r in useless_i if r not in useful_i]
    print 'filename is ', filename
    print ' \tuser_part\titem_part'
    print 'important\t%s\t%s' % (','.join(pretty_order(useful_u)), ','.join(pretty_order(useful_i)))
    print 'useless\t%s\t%s' % (','.join(pretty_order(useless_u)), ','.join(pretty_order(useless_i)))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        run(filename)
    else:
        print 'please specify the res filename of W or V'
