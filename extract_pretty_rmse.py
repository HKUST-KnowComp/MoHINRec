#coding=utf8
'''
    extract res for filling in the excel
'''
import sys
import numpy as np

dt = 'ciaodvd'
motif = sys.argv[1]
print 'extract the result of ', motif

filename = 'exp_res/%s_new_test_res.txt' % dt
split_val_rmses = []
split_test_rmses = []
all_lines = open(filename, 'r').readlines()
for split in range(1,4):
    split_str = 'split%s' % split
    reg2val_rmse = {}
    reg2test_rmse = {}
    lines = [l for l in all_lines if split_str in l]
    for l in lines:
        parts = l.strip().split(',')
        if parts[4] != motif:
            continue
        reg = float(parts[1])
        val_rmse = float(parts[6])
        test_rmse = float(parts[-2])
        alpha = float(parts[5])
        reg2val_rmse.setdefault(reg, []).append((alpha, val_rmse))
        reg2test_rmse.setdefault(reg, []).append((alpha, test_rmse))

    print split_str, ' validation error'
    reg2val_rmse = sorted(reg2val_rmse.items(), key=lambda d:d[0])
    print reg2val_rmse
    for reg, rmses in reg2val_rmse:
        print_rmses = sorted(rmses, key = lambda d:d[0])
        print '\t'.join([str(round(r[1], 4)) for r in print_rmses])
        split_val_rmses.append([round(r[1], 4) for r in print_rmses])


    print split_str, ' test error'
    reg2test_rmse = sorted(reg2test_rmse.items(), key=lambda d:d[0])
    print reg2test_rmse
    for reg, rmses in reg2test_rmse:
        print_rmses = sorted(rmses, key = lambda d:d[0])
        print '\t'.join([str(round(r[1], 4)) for r in print_rmses])
        split_test_rmses.append([round(r[1], 4) for r in print_rmses])

#print the avg of multiple splits
val_rmses = np.asarray(split_val_rmses)
M,N = val_rmses.shape
avg_val_rmses = np.zeros((M/3, N))
for rnd in range(3):
    start, end = rnd * 5, (rnd+1)*5
    avg_val_rmses += val_rmses[start:end]
avg_val_rmses /= 3
print 'avg val error'
for rmses in avg_val_rmses:
    print '\t'.join([str(round(r, 4)) for r in rmses])

test_rmses = np.asarray(split_test_rmses)
avg_test_rmses = np.zeros((M/3, N))
for rnd in range(3):
    start, end = rnd * 5, (rnd+1)*5
    avg_test_rmses += test_rmses[start:end]
avg_test_rmses /= 3
print 'avg test error'
for rmses in avg_test_rmses:
    print '\t'.join([str(round(r, 4)) for r in rmses])

