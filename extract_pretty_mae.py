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
split_val_maes = []
split_test_maes = []
all_lines = open(filename, 'r').readlines()
for split in range(1,4):
    split_str = 'split%s' % split
    reg2val_mae = {}
    reg2test_mae = {}
    lines = [l for l in all_lines if split_str in l]
    for l in lines:
        parts = l.strip().split(',')
        if parts[4] != motif:
            continue
        reg = float(parts[1])
        val_mae = float(parts[7])
        test_mae = float(parts[-1])
        alpha = float(parts[5])
        reg2val_mae.setdefault(reg, []).append((alpha, val_mae))
        reg2test_mae.setdefault(reg, []).append((alpha, test_mae))

    print split_str, ' validation error'
    reg2val_mae = sorted(reg2val_mae.items(), key=lambda d:d[0])
    print reg2val_mae
    for reg, maes in reg2val_mae:
        print_maes = sorted(maes, key = lambda d:d[0])
        print '\t'.join([str(round(r[1], 4)) for r in print_maes])
        split_val_maes.append([round(r[1], 4) for r in print_maes])


    print split_str, ' test error'
    reg2test_mae = sorted(reg2test_mae.items(), key=lambda d:d[0])
    print reg2test_mae
    for reg, maes in reg2test_mae:
        print_maes = sorted(maes, key = lambda d:d[0])
        print '\t'.join([str(round(r[1], 4)) for r in print_maes])
        split_test_maes.append([round(r[1], 4) for r in print_maes])

#print the avg of multiple splits
val_maes = np.asarray(split_val_maes)
M,N = val_maes.shape
avg_val_maes = np.zeros((M/3, N))
for rnd in range(3):
    start, end = rnd * 5, (rnd+1)*5
    avg_val_maes += val_maes[start:end]
avg_val_maes /= 3
print 'avg val error'
for maes in avg_val_maes:
    print '\t'.join([str(round(r, 4)) for r in maes])

test_maes = np.asarray(split_test_maes)
avg_test_maes = np.zeros((M/3, N))
for rnd in range(3):
    start, end = rnd * 5, (rnd+1)*5
    avg_test_maes += test_maes[start:end]
avg_test_maes /= 3
print 'avg test error'
for maes in avg_test_maes:
    print '\t'.join([str(round(r, 4)) for r in maes])

