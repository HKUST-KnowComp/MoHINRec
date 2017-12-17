#!/project/edbg/data/anaconda2/bin/python
#coding=utf8
'''
    compute the sparsity of W and V given the corresponding log files
'''
import sys
import re

import numpy as np

def get_sparsity(w_resfilename, v_resfilename):
    '''
        given the res filenames of W and V, compute the overall sparsity
    '''
    W = np.loadtxt(w_resfilename)
    V = np.loadtxt(v_resfilename)
    sparsity = 1 - (np.count_nonzero(W) + np.count_nonzero(V)) * 1.0 / (W.size + V.size)
    return sparsity

def extract_from_log(filename):
    '''
        given the log filename, extract the reg, saved W and V
    '''
    lines = open(filename, 'r').readlines()
    p1 = re.compile('\'reg_W\': (.+)}')
    p2 = re.compile('W and P saved in (.+) and (.+)')
    p3 = re.compile('\'max_iters\': (\d+?),')
    res = []
    for ind, l in enumerate(lines):
        if 'fm_anova_kernel_glasso finish' in l:
            config_str = lines[ind - 2].strip()
            #import pdb;pdb.set_trace()
            reg = float(p1.search(config_str).group(1))
            ite = int(p3.search(config_str).group(1))
            if ite == 1000:
                continue
            res_str = lines[ind - 4].strip()
            w_resfilename = p2.search(res_str).group(1)
            v_resfilename = p2.search(res_str).group(2)
            sparsity = get_sparsity(w_resfilename, v_resfilename)
            res.append((reg, sparsity, w_resfilename, v_resfilename, filename))
    return res

def run(dt):
    if dt == 'yelp':
        log_filenames = ['log/yelp-200k_all_fm_non_con_reg_reg1e-05.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.01.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.02.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.03.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.04.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.05.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.06.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.07.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.08.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.09.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.1.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.2.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.3.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.4.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg0.5.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg1.0.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg10.0.log',
                         'log/yelp-200k_all_fm_non_con_reg_reg100.0.log',
                         ]
    elif dt == 'amazon':
        log_filenames = ['log/amazon-200k_all_fm_non_con_reg_reg1e-05.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.0001.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.01.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.02.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.03.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.04.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.05.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.06.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.07.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.08.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.09.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.1.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.2.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.3.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.4.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg0.5.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg1.0.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg10.0.log',
                         'log/amazon-200k_all_fm_non_con_reg_reg100.0.log',
                         ]
    res = []
    for filename in log_filenames:
        try:
            res.extend(extract_from_log(filename))
        except Exception:
            continue
    res = sorted(res, key=lambda d: d[0])
    for r in res:
        print '%s\t%.3f\t%s\t%s\t%s' % r

if __name__ == '__main__':
    if len(sys.argv) == 2:
        dt = sys.argv[1]
        run(dt)
