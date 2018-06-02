#coding=utf8
'''
    extract validation error from logfile
'''
import os
import re
dt = 'epinions'

filenames = [r for r in os.listdir('log/') if '%s_motif_fnorm_m' % dt in r]
res = []
for reg in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
    sfs = [r for r in filenames if 'reg%s' % reg in r]
    for filename in sfs:
        lines = open('log/%s' % filename, 'r').readlines()
        for ind, l in enumerate(lines):
            if 'fm_anova_kernel finish' in l:
                wpf_line = lines[ind - 4]
                parts = wpf_line.split()
                wf, pf = parts[6], parts[8]
                config_line = lines[ind - 2]
                split_num = re.search('(split\d)', wf).group(1)
                #print filename, ind, config_line
                mg = re.search("meta_graphs': \[(.*)\]", config_line).group(1)
                motif = re.search('UUB_(m\d)', mg).group(1)
                alpha  = re.search("UUB_m\d_(.*?)',", mg).group(1)
                rmse_line = lines[ind + 2]
                rmse = re.search("rmse=(.*?),", rmse_line).group(1)
                mae = re.search("mae=(.*)", rmse_line).group(1)
                res.append((str(split_num), str(reg), mg, motif, alpha, rmse, mae, wf, pf))

res = sorted(res, key=lambda x: (x[0], x[1], x[3], x[4]))
wfilename = 'exp_res/%s_new_res.txt' % dt
res = [','.join(r) for r in res]
fw = open(wfilename, 'w+')
fw.write('\n'.join(res))
fw.close()
print 'save %s entries in %s' % (len(res), wfilename)
