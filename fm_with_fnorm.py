#coding=utf8
'''
    Implement the solver factorization machine with frobenius norm
'''
import sys
import time
from datetime import datetime
import cPickle as pickle
import logging
import threading

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix as csr

from logging_util import init_logger

DEBUG = False
INCLUDE_RATINGS = True

def init_conifg(dt_arg, reg, exp_type, eps_str=''):
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
        log_filename = 'log/%s_fm_fnorm_once_reg%s.log' % (dt, reg)
    elif eps_str:
        log_filename = 'log/%s_fm_fnorm_reg%s_%s.log' % (dt, reg, eps_str)
    else:
        log_filename = 'log/%s_fm_fnorm_reg%s.log' % (dt, reg)
    exp_id = int(time.time())
    logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)

def cal_err(X, Y, W, V, b):
    part1 = np.dot(W, X.T)
    part2 = np.square(np.dot(X, V))
    part3 = np.dot(np.square(X), np.square(V))
    Y_t = b + part1 + 0.5 * (part2 - part3).sum(axis=1)
    return Y_t - Y

def obj(err, lamb, W, V, b):
    return np.power(err, 2).sum() + lamb * (np.power(norm(W), 2) + np.power(norm(V), 2) + b*b)

def load_data(t_dir, N, train_filename, test_filename):
    start_time = time.time()

    train_data = np.loadtxt(train_filename)
    test_data = np.loadtxt(test_filename)
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]

    uid2reps, bid2reps = load_representation(t_dir, N/2)

    X = np.zeros((train_num, N))
    Y = train_data[:,2]
    test_X = np.zeros((len(test_data), N))
    test_Y = test_data[:,2]

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
    print 'train_data: (%.4f,%.4f), test_data: (%.4f,%.4f)' % (np.mean(Y), np.std(Y), np.mean(test_Y), np.std(test_Y))

    logger.info('finish loading data, ratings_file=%s, cost %.2f seconds, stat(shape, sparsity): train: (%s, %.4f), test: (%s, %.4f)', rating_filename, time.time() - start_time, X.shape, X_sparsity, test_X.shape, test_X_sparsity)

    return X,Y,test_X,test_Y

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
        ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBStarsB_top1000_user.dat', 'UPBStateB_top1000_user.dat', 'UPBCityB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'UUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
        vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBStarsB_top1000_item.dat', 'UPBStateB_top1000_item.dat', 'UPBCityB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'UUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']
    elif dt == 'amazon-200k':
        ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBBrandB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
        vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBBrandB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']

    if INCLUDE_RATINGS:
        ufiles.append('ratings_only_user.dat')
        vfiles.append('ratings_only_item.dat')

    logger.info('run for all, len(ufiles)=%s, len(vfiles)=%s, ufiles=%s, vfiles=%s', len(ufiles), len(vfiles), '|'.join(ufiles), '|'.join(vfiles))

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

def load_data_from_ratings():
    '''
        load data from raw ratings, then it can be a very sparse matrix
    '''
    train_filename = '../libfm/data/ml/ml-100k-rating_train_1.txt.libfm'
    test_filename = '../libfm/data/ml/ml-100k-rating_test_1.txt.libfm'
    N = 2625

    lines = open(train_filename, 'r').readlines()
    M = len(lines)
    rows, cols = [], []
    Y = []
    for ind, l in enumerate(lines):
        parts = l.strip().split()
        Y.append(float(parts[0]))
        for p in parts[1:]:
            col,_ = p.split(':')
            rows.append(ind)

            cols.append(int(col))

    X = csr(([1] * M * 2, (rows, cols)), shape=(M,N))

    lines = open(test_filename, 'r').readlines()
    M = len(lines)
    rows, cols = [], []
    test_Y = []
    for ind, l in enumerate(lines):
        parts = l.strip().split()
        test_Y.append(float(parts[0]))
        for p in parts[1:]:
            col,_ = p.split(':')
            rows.append(ind)
            cols.append(int(col))

    test_X = csr(([1] * M * 2, (rows, cols)), shape=(M,N))
    X = X.toarray()
    Y = np.array(Y)
    test_X = test_X.toarray()
    test_Y = np.array(test_Y)
    return X, Y, test_X, test_Y

def cal_rmse(W, V, b, test_X, test_Y):
    err = cal_err(test_X, test_Y, W, V, b)
    num = test_Y.shape[0]
    rmse = np.sqrt(np.square(err).sum() / num)
    return rmse

def cal_mae(W, V, b, test_X, test_Y):
    err = cal_err(test_X, test_Y, W, V, b)
    num = test_Y.shape[0]
    mae = np.abs(err).sum() / num
    return mae

def run(split_num, t_dir, lamb, K, eps, ite, train_filename, test_filename):
    '''
        基于block进行更新，同一个block内部的interaction为0
    '''
    if dt == 'yelp-200k':
        L = 11
    elif dt == 'amazon-200k':
        L = 8
    else:
        L = 8
    F = 10
    if INCLUDE_RATINGS:
        L += 1
    N = 2 * L * F
    N = 2625
    eta = 1e-6#learning rate
    beta = 0.9#parameter used in line search
    exp_id = int(time.time())

    b = 0.0
    W = np.random.rand(N) * 0.00001# 1 by N
    V = np.random.rand(N, K) * 0.00001# N by K
    #X, Y, test_X, test_Y = load_data(t_dir, N, train_filename, test_filename)
    X, Y, test_X, test_Y = load_data_from_ratings()

    exp_info = 'test W,V on %s(real data),  when reg=%s' % (dt, lamb)
    exp_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    logger.info('*************exp_time:%s, exp_id=%s, %s*************', exp_time, exp_id, exp_info)
    logger.info('fm_with_fnorm started, exp_id=%s, %s group features, config K,reg,ite,eta=(%s,%s,%s,%s)', int(exp_id), L, K, lamb, ite, eta)

    err = cal_err(X, Y, W, V, b)
    objs = [obj(err, lamb, W, V, b)]
    rmses = [cal_rmse(W, V, b, test_X, test_Y)]
    maes = [cal_mae(W, V, b, test_X, test_Y)]

    start = time.time()
    lt = 0
    ln = 1000
    for t in range(ite):
        start = time.time()
        #cal gradients
        b = b - eta * 2 * (err.sum() + lamb * b)
        grad_w = 2 * np.dot(err, X) + 2 * lamb * W #element-wise correspondence
        part = np.dot(X, V)
        grad_v = np.zeros(V.shape)
        for i in range(N):
            tmp = np.square(X[:,i])
            for f in range(K):
                grad_v[i,f] = 2 * np.dot(err, np.multiply(X[:,i], part[:,f]) - V[i,f] * tmp)
        grad_v += 2 * lamb * V

        #line search
        if DEBUG:
            print 'start line search...'
        for lt in range(ln+1):
            tW, tV = W - eta * grad_w, V - eta * grad_v
            err = cal_err(X, Y, tW, tV, b)
            obj_p = obj(err, lamb, tW, tV, b)
            if DEBUG:
                print 'lt=%s, obj_p=%s' % (lt, obj_p)
            if obj_p < objs[t]:
                objs.append(obj_p)
                W, V = tW, tV
                eta = 1.1 * eta
                break
            else:
                eta = beta * eta

        rmses.append(cal_rmse(W, V, b, test_X, test_Y))
        maes.append(cal_mae(W, V, b, test_X, test_Y))
        end = time.time()

        if lt == ln:
            logger.info('lt=%s', lt)
            break

        dr = abs(objs[t] - objs[t+1]) / objs[t]
        logger.info('iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
        if  dr < eps:
            break
    logger.info('fm_with_fnorm finished, exp_time=%s, exp_id=%s, %s group features, config K,reg,ite,eta=(%s,%s,%s,%s)', exp_time, int(exp_id), L, K,lamb,ite,eta)
    return rmses, maes

def run_acc(split_num, t_dir, lamb, K, eps, ite, train_filename, test_filename):
    '''
        基于block进行更新，同一个block内部的interaction为0
    '''
    global exp_rmses
    global exp_maes
    global threads_finish
    logger.info('start validation %s, exp_dir=%s, train_filename=%s, test_filename=%s', split_num, t_dir, train_filename, test_filename)
    start_time = time.time()
    if dt == 'yelp-200k':
        L = 11
    if dt == 'yelp-200k':
        L = 11
    elif dt == 'amazon-200k':
        L = 8
    else:
        L = 8
    F = 10
    if INCLUDE_RATINGS:
        L += 1
    N = 2 * L * F
    N = 2625
    eta = 1e-7#learning rate
    beta = 0.9#parameter used in line search
    exp_id = int(time.time())

    b = 0.0
    W = np.random.rand(N) * 0.00001# 1 by N
    V = np.random.rand(N, K) * 0.00001# N by K
    #X, Y, test_X, test_Y = load_data(t_dir, N, train_filename, test_filename)
    X, Y, test_X, test_Y = load_data_from_ratings()

    exp_info = 'test W,V on %s(real data),  when reg=%s' % (dt, lamb)
    exp_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    logger.info('*************exp_time:%s, exp_id=%s, %s*************', exp_time, exp_id, exp_info)
    logger.info('fm_with_fnorm started, exp_id=%s, %s group features, config K,reg,ite,eta=(%s,%s,%s,%s)', int(exp_id), L, K, lamb, ite, eta)

    objs = [None] * (ite + 1)
    err = cal_err(X, Y, W, V, b)
    objs[0] = obj(err, lamb, W, V, b)
    rmses = [cal_rmse(W, V, b, test_X, test_Y)]
    maes = [cal_mae(W, V, b, test_X, test_Y)]

    start = time.time()
    lt = 0
    ln = 1000

    A = np.hstack((W.reshape(-1,1), V))
    A0, A1, C1 = A.copy(), A.copy(), A.copy()
    c = objs[0]
    r0, r1, q, qeta = 0.0, 1.0, 1.0, 0.5
    eta1 = eta2 = eta

    lt1, lt2 = 0, 0
    ln = 1000
    for t in range(ite):
        start = time.time()
        B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
        W, V = B[:,0].flatten(), B[:,1:]

        err = cal_err(X, Y, W, V, b)
        obj_b = obj(err, lamb, W, V, b)

        b = b - eta * 2 * (err.sum() + lamb * b)

        l1start = time.time()
        eta1, lt1, obj_c, W_p, V_p, v_cost1 = line_search(err, W, V, b, lamb, eta1, obj_b, X, Y, ln, N, K)
        l1cost = time.time() - l1start

        if lt1 == ln:
            logger.info('lt1=%s', lt1)
            break

        C1 = np.hstack((W_p.reshape(-1,1), V_p))
        A0 = A1.copy()

        l2cost, v_cost2 = 0.0, 0.0

        if obj_c < c:
            A1 = C1.copy()
            objs[t+1] = obj_c
        else:
            W, V = A1[:,0].flatten(), A1[:,1:]
            err = cal_err(X, Y, W, V, b)
            obj_a = obj(err, lamb, W, V, b)

            l2start = time.time()
            eta2, lt2, obj_v, W_p, V_p, v_cost2 = line_search(err, W, V, b, lamb, eta2, obj_a, X, Y, ln, N, K)
            l2cost = time.time() - l2start

            if obj_c > obj_v:
                A1 = np.hstack((W_p.reshape(-1,1), V_p))
                objs[t+1] = obj_v
            else:
                A1 = C1.copy()
                objs[t+1] = obj_c

        if lt2 == ln:
            logger.info('lt2=%s', lt2)
            break

        W, V = A1[:,0].flatten(), A1[:,1:]

        rmses.append(cal_rmse(W, V, b, test_X, test_Y))
        maes.append(cal_mae(W, V, b, test_X, test_Y))
        end = time.time()

        dr = abs(objs[t] - objs[t+1]) / objs[t]
        logger.info('exp_id=%s, iter=%s, (lt1,eta1, v_cost1/cost)=(%s,%s, %.2f/%.2fs), (lt2,eta2,v_cost2/cost)=(%s,%s, %.2f/%.2fs), obj=%.5f(dr=%.8f), rmse=%.5f, mae=%.5f, cost=%.2f seconds', exp_id, t, lt1, eta1, v_cost1, l1cost, lt2, eta2, v_cost2, l2cost, objs[t+1], dr, rmses[t+1], maes[t+1], (end - start))

        r0 = r1
        r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0
        tq = qeta * q + 1.0
        c = (qeta * q * c + objs[t+1]) / tq
        q = tq

        if  dr < eps:
            break
    logger.info('fm_with_fnorm finished, exp_time=%s, exp_id=%s, %s group features, config K,reg,ite,eta=(%s,%s,%s,%s)', exp_time, int(exp_id), L, K,lamb,ite,eta)
    #round_rmse = np.mean(rmses[-5:])
    #round_mae = np.mean(maes[-5:])
    #exp_rmses[split_num] = round_rmse
    #exp_maes[split_num] = round_mae
    #logger.info('finish validation %s, exp_dir=%s, cost %.2f minutes, rmse=%.4f, mae=%.4f', split_num, t_dir, (time.time() - start_time) / 60.0, exp_rmses[split_num], exp_maes[split_num])
    #threads_finish[split_num - 1] = True
    #logger.info('finish validation %s, exp_dir=%s, cost %.2f minutes, rmse=%.4f, mae=%.4f', split_num, t_dir, (time.time() - start_time) / 60.0, exp_rmses[split_num], exp_maes[split_num])
    return rmses, maes

def line_search(err, W, V, b, lamb, eta, obj_v, X, Y, ln, N, K):

    grad_start = time.time()
    grad_w = 2 * np.dot(err, X) + 2 * lamb * W
    w_cost = time.time() - grad_start
    part = np.dot(X, V).copy(order='F')
    grad_v = np.zeros(V.shape,dtype=np.float64, order='F')
    v_start = time.time()
    for i in range(N):
        tmp = np.square(X[:,i])
        for f in range(K):
            k_start = time.time()
            #grad_v[i,f] = cal_grad_v_by_c(err, X[:,i], part[:,f], V[i,f] * tmp)
            grad_v[i,f] = 2 * np.dot(err, np.multiply(X[:,i], part[:,f]) - V[i,f] * tmp)
    grad_v += 2 * lamb * V
    v_cost = time.time() - v_start

    #line search with accelerated proximal operator
    for lt in range(ln+1):
        W_p, V_p = W - eta * grad_w, V - eta * grad_v
        l_err = cal_err(X, Y, W_p, V_p, b)
        l_obj = obj(l_err, lamb, W_p, V_p, b)
        if l_obj < obj_v:
            eta = 1.1 * eta
            break
        else:
            eta = 0.9 * eta
    return eta, lt, l_obj, W_p, V_p, v_cost

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
        threads.append(threading.Thread(target=run_acc, args=(rnd+1, t_dir, lamb, K, eps, ite, train_filename, test_filename)))

    for t in threads:
        t.daemon = True
        t.start()

    while True:
        time.sleep(1)
        if sum(threads_finish) == 5:
            cost = (time.time() - run_start) / 60.0
            logger.info('**********finish run_5_validations, cost %.2f mins, dataset=%s,rating_filename=%s***********\n*****config: (reg, K, eps, iters solver)=(%s, %s, %s, %s, %s), exp rmses: %s, maes: %s\n*******avg rmse=%s, avg mae=%s\n**************', cost, dt, rating_filename, lamb, K, eps, ite, solver, exp_rmses.items(), exp_maes.items(), np.mean(exp_rmses.values()), np.mean(exp_maes.values()))
            break

if __name__ == '__main__':
    if len(sys.argv) == 5:
        dt = sys.argv[1]
        lamb = float(sys.argv[3].replace('reg',''))
        exp_type = int(sys.argv[2])
        init_conifg(dt, lamb, exp_type, sys.argv[4])

        exp_id = int(time.time())
        ite = 1000
        K, solver = 100, 'pg'
        eps = float(sys.argv[4].replace('eps',''))
        if int(sys.argv[2]) == 2:
            run_5_validation(lamb, K, eps, ite, solver)
        elif exp_type == 1:
            run_start = time.time()
            split_num = 1
            t_dir = 'data/%s/exp_split/%s/' % (dt, split_num)

            train_filename = t_dir + '%s_train_%s.txt' % (rating_filename, split_num)
            test_filename = t_dir + '%s_test_%s.txt' % (rating_filename, split_num)

            if solver == 'acc':
                rmses, maes = run_acc(split_num, t_dir, lamb, K, eps, ite, train_filename, test_filename)
            else:
                rmses, maes = run(split_num, t_dir, lamb, K, eps, ite, train_filename, test_filename)
            cost = (time.time() - run_start) / 3600.0
            logger.info('**********fm_fnorm finish, run once, cost %.2f hours*******\nconfig: (reg, K, eps, ites, solver)=(%s, %s, %s, %s, %s), rmses: %s, maes: %s\navg rmse=%s, avg mae=%s\n***************', cost, lamb, K, eps, ite, solver, rmses[-5:], maes[-5:], np.mean(rmses[-5:]), np.mean(maes[-5:]))
