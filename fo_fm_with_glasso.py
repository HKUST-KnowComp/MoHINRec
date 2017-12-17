#coding=utf8
'''
    Implement the solver factorization machine with group lasso
    version4: only computing first order FM, meaning that W is the parameters
    version3: gradient calculation with c code for V
    version2: accelerated proximal gradient descent
    version1: proximal gradient descent
'''
import sys
import time
import logging
import ctypes
import threading
from datetime import datetime
import cPickle as pickle

import numpy as np
from numpy.linalg import norm

from logging_util import init_logger

INCLUDE_RATINGS = True
INCLUDE_RAND = False #add two random path based features to test the effect of group lasso

def init_conifg(dt_arg, reg, exp_type, eps_str='', K=10):
    global rating_filename
    global logger
    global exp_id
    global dt
    dt = dt_arg

    if dt == 'yelp':
        rating_filename = 'ratings_filter5'
    elif dt == 'yelp-200k':
        rating_filename = 'ratings'
    elif dt == 'yelp-sample':
        rating_filename = ''
    elif dt in ['ml-100k', 'ml-1m', 'ml-10m']:
        rating_filename = '%s-rating' % dt
    elif dt == 'amazon-app':
        rating_filename = 'filter5_ratings_Apps_for_Android'
    elif dt == 'amazon-200k':
        rating_filename = 'ratings'

    if exp_type == 1:
        log_filename = 'log/%s_fo_fm_glasso_once_reg%s_K%s.log' % (dt, reg, K)
    elif eps_str:
        log_filename = 'log/%s_fo_fm_glasso_reg%s_%s_K%s.log' % (dt, reg, eps_str, K)
        if is_comb:
            log_filename = 'log/%s_fo_fm_glasso_comb_reg%s_%s_K%s.log' % (dt, reg, eps_str, K)
    else:
        log_filename = 'log/%s_fo_fm_glasso_reg%s.log' % (dt, reg)

    if INCLUDE_RAND:
        log_filename = '%s_include_rand.log' % (log_filename.split('.')[0])
    exp_id = int(time.time())
    logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)

stf = lambda b: b if b > 0.0 else 0.0#soft threshold function

DEBUG = False

def prox_op(W, eta, gw_inds):
    eps = 1e-6
    for i in range(len(gw_inds)):
        W[gw_inds[i]] = stf(1 - eta / (norm(W[gw_inds[i]]) + eps)) * W[gw_inds[i]]
    return W

def group_lasso(W, gw_inds):
    res = 0.0
    for i in range(len(gw_inds)):
        res += norm(W[gw_inds[i]])
    return res

def cal_err(X, Y, W, b):
    part1 = np.dot(W, X.T)
    Y_t = b + part1
    return Y_t - Y

def obj(err, W, lamb, gw_inds):
    return np.power(err, 2).sum() + lamb * group_lasso(W, gw_inds)

def load_data(t_dir, N, train_filename, test_filename):
    start_time = time.time()

    train_data = np.loadtxt(train_filename)
    test_data = np.loadtxt(test_filename)
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]

    uid2reps, bid2reps = load_representation(t_dir, N/2)

    X = np.zeros((train_num, N), dtype=np.float64, order='F')
    Y = train_data[:,2].copy(order='F')
    test_X = np.zeros((test_num, N), dtype=np.float64, order='F')
    test_Y = test_data[:,2].copy(order='F')

    ind = 0
    for u, b, _ in train_data:
        ur = uid2reps[int(u)]
        br = bid2reps[int(b)]
        X[ind] = np.concatenate((ur,br))
        ind += 1
    X_sparsity = np.count_nonzero(X) * 1.0 / X.size

    ind = 0
    for u, b, _ in test_data:
        ur = uid2reps.get(int(u), np.zeros(N/2, dtype=np.float64))
        br = bid2reps.get(int(b), np.zeros(N/2, dtype=np.float64))
        test_X[ind] = np.concatenate((ur,br))
        ind += 1

    test_X_sparsity = np.count_nonzero(test_X) * 1.0 / test_X.size

    logger.info('finish loading data, cost %.2f seconds, ratings_file=%s, train=%s, test=%s, stat(shape, sparsity): train: (%s, %.4f), test: (%s, %.4f)', time.time() - start_time, rating_filename, train_filename, test_filename, X.shape, X_sparsity, test_X.shape, test_X_sparsity)
    return X, Y, test_X, test_Y

def load_representation(t_dir, fnum):
    '''
        load user and item latent features generate by MF for every meta-graph
    '''
    if dt in ['yelp-200k', 'amazon-200k']:
        ufilename = t_dir + 'uids.txt'
        bfilename = t_dir + 'bids.txt'
    uids = [int(l.strip()) for l in open(ufilename, 'r').readlines()]
    uid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in uids}

    bids = [int(l.strip()) for l in open(bfilename, 'r').readlines()]
    bid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in bids}

    if dt == 'yelp-200k':
        ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top100_user.dat', 'UPBStarsB_top100_user.dat', 'UPBStateB_top100_user.dat', 'UPBCityB_top100_user.dat', 'UPBUB_top100_user.dat', 'UNBUB_top100_user.dat', 'UUB_top100_user.dat', 'URPARUB_top100_user.dat', 'URNARUB_top100_user.dat']
        vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top100_item.dat', 'UPBStarsB_top100_item.dat', 'UPBStateB_top100_item.dat', 'UPBCityB_top100_item.dat', 'UPBUB_top100_item.dat', 'UNBUB_top100_item.dat', 'UUB_top100_item.dat', 'URPARUB_top100_item.dat', 'URNARUB_top100_item.dat']
    elif dt == 'amazon-200k':
        ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBBrandB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
        vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBBrandB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']

    if is_comb:
        ufiles, vfiles = [], []
        path_strs = ['UBUB', 'URARUB']
        combs = ['PPP', 'NNP', 'PPN', 'NNN', 'PNP', 'NPP', 'PNN', 'NPN']
        for ps in path_strs:
            for comb in combs:
                ufiles.append('combs/%s_%s_top50_user.res' % (ps, comb))
                vfiles.append('combs/%s_%s_top50_item.res' % (ps, comb))

    if INCLUDE_RATINGS:
        ufiles.append('ratings_only_user.dat')
        vfiles.append('ratings_only_item.dat')

    logger.info('run for %s, len(ufiles)=%s, len(vfiles)=%s, ufiles=%s, vfiles=%s', 'comb' if is_comb else 'all', len(ufiles), len(vfiles), '|'.join(ufiles), '|'.join(vfiles))

    for find, filename in enumerate(ufiles):
        ufs = np.loadtxt(t_dir + 'mf_features/path_count/' + filename, dtype=np.float64)
        cur = find * 10
        for uf in ufs:
            uid = int(uf[0])
            f = uf[1:]
            uid2reps[uid][cur:cur+10] = f

    for find, filename in enumerate(vfiles):
        bfs = np.loadtxt(t_dir + 'mf_features/path_count/' + filename, dtype=np.float64)
        cur = find * 10
        for bf in bfs:
            bid = int(bf[0])
            f = bf[1:]
            bid2reps[bid][cur:cur+10] = f
    return uid2reps, bid2reps

def cal_rmse(W, b, test_X, test_Y):
    err = cal_err(test_X, test_Y, W, b)
    num = test_Y.shape[0]
    rmse = np.sqrt(np.square(err).sum() / num)
    return rmse

def cal_mae(W, b, test_X, test_Y):
    err = cal_err(test_X, test_Y, W, b)
    num = test_Y.shape[0]
    mae = np.abs(err).sum() / num
    return mae

def run(split_num, t_dir, lamb, K, eps, ite, solver='acc', train_filename='', test_filename=''):
    '''
        All the arrays are adoping the column-layout memory schema, controlled by the parameter order'F', which is for the C code calculation of gradients of V
        K: number of latent features in FM
        lamb: regularization
        ite: max iterations regardless of stopping criteira
        eps: stopping criteria
        eta: learning rate
        F: number of latent features in matrix factorization
        L: number of meta-graph
    '''
    global exp_rmses
    global exp_maes
    global threads_finish
    logger.info('start validation %s, exp_dir=%s, train_filename=%s, test_filename=%s', split_num, t_dir, train_filename, test_filename)
    start_time = time.time()
    if dt == 'yelp-200k':
        L = 11
    elif dt == 'amazon-200k':
        L = 8
    if is_comb:
        L = 16
    F = 10
    if INCLUDE_RATINGS:
        L += 1
    if INCLUDE_RAND:
        L += 2
    N = 2 * L * F
    eta = 1e-7#learning rate
    beta = 0.9#parameter used in line search
    exp_id = int(time.time())

    b = 0.0 # bias
    W = np.random.rand(N).astype(dtype=np.float64, order='F') * 0.0001 # 1 by N
    X, Y, test_X, test_Y = load_data(t_dir, N, train_filename, test_filename)

    exp_info = 'exp on large scale data, 1-5 scale, op use lamb * eta, when reg=%s, solver=%s' % (lamb, solver)
    exp_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    logger.info('*************exp_time:%s, exp_id=%s, %s*************', exp_time, exp_id, exp_info)
    logger.info('fo_fm_group_lasso started, exp_id=%s, %s group features, solver=%s, config K,reg,ite,eta,eps=(%s,%s,%s,%s, %s)', int(exp_id), L, solver, K, lamb, ite, eta, eps)

    gw_inds = np.arange(N).reshape(2*L, F, order='F')

    if solver == 'acc':
        rmses, maes = train_acc_prox_gradient(t_dir, X, Y, W, b, gw_inds, test_X, test_Y, ite, eta, beta, eps, exp_id, N, K)
    else:
        rmses, maes = train_prox_gradient(X, Y, W, b, gw_inds, test_X, test_Y, ite, eta, beta, eps, exp_id, N, K)

    total_cost = (time.time() - exp_id) / 3600.0
    logger.info('fo_fm_group_lasso finished, total_cost=%.2f hours exp_time=%s, exp_id=%s, %s group features, solver=%s, config K,reg,ite,eta, eps=(%s,%s,%s,%s,%s)', total_cost, exp_time, int(exp_id), L, solver, K, lamb,ite,eta, eps)
    round_rmse = np.mean(rmses[-5:])
    round_mae = np.mean(maes[-5:])
    exp_rmses[split_num] = round_rmse
    exp_maes[split_num] = round_mae
    logger.info('finish validation %s, exp_dir=%s, cost %.2f minutes, rmse=%.4f, mae=%.4f', split_num, t_dir, (time.time() - start_time) / 60.0, exp_rmses[split_num], exp_maes[split_num])
    threads_finish[split_num - 1] = True
    return rmses, maes

def train_prox_gradient(X, Y, W, b, gw_inds, test_X, test_Y, ite, eta, beta, eps, exp_id, N, K):
    err = cal_err(X, Y, W, b)
    obj = np.power(err, 2).sum() + lamb * group_lasso(W, gw_inds)
    objs = [obj]
    rmses = [cal_rmse(W, b, test_X, test_Y)]
    maes = [cal_mae(W, b, test_X, test_Y)]

    start = time.time()
    lt = 0
    ln = 1000
    for t in range(ite):
        start = time.time()
        #cal gradients
        b = b - eta * 2 * err.sum()
        grad_w = 2 * np.dot(err, X)#element-wise correspondence

        #line search with proximal operator
        if DEBUG:
            print 'start line search...'
        for lt in range(ln+1):
            tW = W - eta * grad_w
            W_p = prox_op(tW, eta * lamb, gw_inds)
            err = cal_err(X, Y, W_p, b)
            obj_p = np.power(err, 2).sum() + lamb * group_lasso(W_p, gw_inds)
            if DEBUG:
                print 'lt=%s, obj_p=%s' % (lt, obj_p)
            if obj_p < objs[t]:
                objs.append(obj_p)
                W  = W_p
                eta = 1.1 * eta
                break
            else:
                eta = beta * eta

        rmses.append(cal_rmse(W, b, test_X, test_Y))
        maes.append(cal_mae(W, b, test_X, test_Y))
        end = time.time()

        if lt == ln:
            print 'lt=%s' % lt
            break

        dr = abs(objs[t] - objs[t+1]) / objs[t]
        logger.info('exp_id=%s, iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', exp_id, t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
        if  dr < eps:
            break
    return rmses, maes

def line_search(err, W, b, lamb, eta, gw_inds, obj_v, X, Y, ln, N, K):

    grad_start = time.time()
    grad_w = 2 * np.dot(err, X)
    w_cost = time.time() - grad_start

    #print 'grad_w/square/set/grads cost: %.2fs/%.2fs/%.2fs/%.2fs' % (w_cost, square_cost, v_set_cost, time.time() - grad_start)
    #line search with accelerated proximal operator
    for lt in range(ln+1):
        tW = W - eta * grad_w
        W_p = prox_op(tW, eta * lamb, gw_inds)
        l_err = cal_err(X, Y, W_p, b)
        l_obj = obj(l_err, W_p, lamb, gw_inds)
        if l_obj < obj_v:
            eta = 1.1 * eta
            break
        else:
            eta = 0.9 * eta
    return eta, lt, l_obj, W_p

def train_acc_prox_gradient(t_dir, X, Y, W, b, gw_inds, test_X, test_Y, ite, eta, beta, eps, exp_id, N, K):
    '''
        accelerated proximal gradient method
    '''
    objs = [None] * (ite + 1)
    err = cal_err(X, Y, W, b)
    objs[0] = obj(err, W, lamb, gw_inds)
    rmses = [cal_rmse(W, b, test_X, test_Y)]
    maes = [cal_mae(W, b, test_X, test_Y)]

    A = W.copy()
    A0, A1, C1 = A.copy(), A.copy(), A.copy()
    c = objs[0]
    r0, r1, q, qeta = 0.0, 1.0, 1.0, 0.5
    eta1 = eta2 = eta

    lt1, lt2 = 0, 0
    ln = 1000
    for t in range(ite):
        start = time.time()
        B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
        W = B.copy()

        err = cal_err(X, Y, W, b)
        obj_b = obj(err, W, lamb, gw_inds)

        b = b - eta * 2 * err.sum()

        l1start = time.time()
        eta1, lt1, obj_c, W_p = line_search(err, W, b, lamb, eta1, gw_inds, obj_b, X, Y, ln, N, K)
        l1cost = time.time() - l1start

        if lt1 == ln:
            logger.info('lt1=%s', lt1)
            break

        #C1 = np.hstack((W_p.reshape(-1,1), V_p))
        C1 = W_p.copy()
        A0 = A1.copy()

        l2cost = 0.0

        if obj_c < c:
            A1 = C1.copy()
            objs[t+1] = obj_c
        else:
            W = A1.copy()
            err = cal_err(X, Y, W, b)
            obj_a = obj(err, W, lamb, gw_inds)

            l2start = time.time()
            eta2, lt2, obj_v, W_p = line_search(err, W, b, lamb, eta2, gw_inds, obj_a, X, Y, ln, N, K)
            l2cost = time.time() - l2start

            if obj_c > obj_v:
                A1 = W_p.copy()
                objs[t+1] = obj_v
            else:
                A1 = C1.copy()
                objs[t+1] = obj_c

        if lt2 == ln:
            logger.info('lt2=%s', lt2)
            break

        W = A1.copy()

        rmses.append(cal_rmse(W, b, test_X, test_Y))
        maes.append(cal_mae(W, b, test_X, test_Y))
        end = time.time()

        dr = abs(objs[t] - objs[t+1]) / objs[t]
        logger.info('exp_id=%s, iter=%s, (lt1,eta1, cost)=(%s,%s, %.2fs), (lt2,eta2,cost)=(%s,%s, %.2fs), obj=%.5f(dr=%.8f), rmse=%.5f, mae=%.5f, cost=%.2f seconds', exp_id, t, lt1, eta1, l1cost, lt2, eta2, l2cost, objs[t+1], dr, rmses[t+1], maes[t+1], (end - start))

        r0 = r1
        r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0
        tq = qeta * q + 1.0
        c = (qeta * q * c + objs[t+1]) / tq
        q = tq

        if  dr < eps:
            break

    split_num = t_dir.split('/')[-2]
    W_wfilename = 'fm_res/split%s_W_%s_exp%s.txt' % (split_num, lamb, exp_id)
    np.savetxt(W_wfilename, W)
    logger.info('W saved in %s', W_wfilename)
    return rmses, maes

def run_5_validation(lamb, K, eps, ite, solver):
    logger.info('start run_5_validations, dataset=%s, ratings_filename=%s, K=%s,eps=%s,reg=%s,iters=%s,solver=%s', dt, rating_filename, K,eps,lamb, ite, solver)
    run_start = time.time()
    global exp_rmses
    global exp_maes
    global threads_finish
    exp_maes, exp_rmses = {}, {}
    threads_finish = [False] * 5

    threads = []
    for rnd in xrange(5):
        start_time = time.time()
        t_dir = 'data/%s/exp_split/%s/' % (dt, rnd+1)

        train_filename = t_dir + '%s_train_%s.txt' % (rating_filename, rnd+1)
        test_filename = t_dir + '%s_test_%s.txt' % (rating_filename, rnd+1)

        threads.append(threading.Thread(target=run, args=(rnd+1, t_dir, lamb, K, eps, ite,solver, train_filename, test_filename)))

    for t in threads:
        t.daemon = True
        t.start()

    while True:
        time.sleep(1)
        if sum(threads_finish) == 5:
            cost = (time.time() - run_start) / 60.0
            if INCLUDE_RAND:
                logger.info('**********************TWO RANDOM GROUP OF FEATURES ARE USED**************************')
            logger.info('**********finish run_5_validations, cost %.2f mins, dataset=%s,rating_filename=%s***********\n*****config: (reg, K, eps, iters solver)=(%s, %s, %s, %s, %s), exp rmses: %s, maes: %s\n*******avg rmse=%s, avg mae=%s\n**************', cost, dt, rating_filename, lamb, K, eps, ite, solver, exp_rmses.items(), exp_maes.items(), np.mean(exp_rmses.values()), np.mean(exp_maes.values()))
            break

if __name__ == '__main__':
    if len(sys.argv) == 7:
        global is_comb
        is_comb = eval(sys.argv[5])

        dt = sys.argv[1]
        lamb = float(sys.argv[3].replace('reg',''))
        exp_type = int(sys.argv[2])
        K = int(sys.argv[6].replace('K',''))

        init_conifg(dt, lamb, exp_type, sys.argv[4], K)


        exp_id = int(time.time())
        ite = 1000
        solver = 'acc'
        eps = float(sys.argv[4].replace('eps',''))
        if exp_type == 1:
            global t_dir
            run_start = time.time()
            split_num = 1
            t_dir = 'data/%s/exp_split/%s/' % (dt, split_num)

            train_filename = t_dir + '%s_train_%s.txt' % (rating_filename, split_num)
            test_filename = t_dir + '%s_test_%s.txt' % (rating_filename, split_num)

            rmses, maes = run(1, lamb, K, eps, ite, solver, train_filename, test_filename)
            cost = (time.time() - run_start) / 3600.0
            logger.info('**********fo_fm_with_group_lasso finish, run once, cost %.2f hours*******\nconfig: (reg, K, eps, ites, solver)=(%s, %s, %s, %s, %s), rmses: %s, maes: %s\navg rmse=%s, avg mae=%s\n***************', cost, lamb, K, eps, ite, solver, rmses[-5:], maes[-5:], np.mean(rmses[-5:]), np.mean(maes[-5:]))
        elif int(sys.argv[2]) == 2:
            run_5_validation(lamb, K, eps, ite, solver)
