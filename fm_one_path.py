#coding=utf8
'''
    code to deal with one-path MF + FM
'''
import sys
import time
import logging
import ctypes
from datetime import datetime
import cPickle as pickle

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix as csr

from logging_util import init_logger

stf = lambda b: b if b > 0.0 else 0.0#soft threshold function

def init_conifg(dt_arg):
    global rating_filename
    global logger
    global exp_id
    global dt
    dt = dt_arg

    if dt in ['yelp-50k', 'yelp-200k']:
        rating_filename = 'ratings'
    elif dt == 'cikm-yelp':
        rating_filename = 'ratings'
    elif dt == 'yelp-sample':
        rating_filename = ''
    elif dt in ['ml-100k', 'ml-1m', 'ml-10m']:
        rating_filename = '%s-rating' % dt
    elif dt == 'amazon-app':
        rating_filename = 'filter5_ratings_Apps_for_Android'
    elif dt == 'amazon-200k':
        rating_filename = 'ratings'

    log_filename = 'log/%s_fm_one_path_%s.log' % (dt, path_str)
    exp_id = int(time.time())
    logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)

DEBUG = False

grad_v_lib = ctypes.cdll.LoadLibrary('./cal_grad_v.so')
cal_grad_v = grad_v_lib.cal_grad_v

def prox_op(W, V, eta, gw_inds, gv_inds):
    eps = 1e-6
    f_V = V.flatten()
    for i in range(len(gw_inds)):
        W[gw_inds[i]] = stf(1 - eta / (norm(W[gw_inds[i]]) + eps)) * W[gw_inds[i]]
        f_V[gv_inds[i]] = stf(1 - eta / (norm(f_V[gv_inds[i]]) + eps)) * f_V[gv_inds[i]]
    V = f_V.reshape(V.shape)
    return W, V

def group_lasso(W, V, gw_inds, gv_inds):
    res = 0.0
    for i in range(len(gw_inds)):
        res += norm(W[gw_inds[i]])
        res += norm(V[gv_inds[i]])
    return res

def cal_err(X, Y, W, V, b):
    part1 = np.dot(W, X.T)
    part2 = np.square(np.dot(X, V))
    part3 = np.dot(np.square(X), np.square(V))
    Y_t = b + part1 + 0.5 * (part2 - part3).sum(axis=1)
    return Y_t - Y

def obj(err, W, V, lamb, gw_inds, gv_inds):
    return np.power(err, 2).sum() + lamb * group_lasso(W, V.flatten(), gw_inds, gv_inds)

def load_data(N, train_filename, test_filename):
    start_time = time.time()

    train_data = np.loadtxt(train_filename)
    test_data = np.loadtxt(test_filename)
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]

    uid2reps, bid2reps = load_representation(N/2)

    X = np.zeros((train_num, N), dtype=np.float64)
    Y = train_data[:,2]
    test_X = np.zeros((test_num, N), dtype=np.float64)
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
        br = bid2reps.get(int(b),np.zeros(N/2, dtype=np.float64))
        test_X[ind] = np.concatenate((ur,br))
        ind += 1

    test_X_sparsity = np.count_nonzero(test_X) * 1.0 / test_X.size

    #import pdb;pdb.set_trace()
    logger.info('finish loading data, ratings_file=%s, cost %.2f seconds, stat(shape, mean, std, sparsity): train_data: (%s, %.4f,%.4f,%.4f), test_data: (%s, %.4f,%.4f,%.4f)', rating_filename, time.time() - start_time, X.shape, np.mean(Y), np.std(Y), X_sparsity, test_X.shape, np.mean(test_Y), np.std(test_Y), test_X_sparsity)
    return X, Y, test_X, test_Y

def load_representation(fnum):
    if dt == 'cikm-yelp':
        ufilename = dir_ + 'uids.txt'
        bfilename = dir_ + 'bids.txt'
    elif dt in ['yelp-200k', 'yelp-50k']:
        ufilename = dir_ + 'uids.txt'
        bfilename = dir_ + 'bids.txt'
        #ufiles = ['ratings_user.dat']
        #vfiles = ['ratings_item.dat']
        #ufiles = ['UPBCatB_top1000_user.dat']
        #vfiles = ['UPBCatB_top1000_item.dat']
        ufiles = ['%s_top1000_user.dat' % path_str]
        vfiles = ['%s_top1000_item.dat' % path_str]
    elif dt == 'ml-1m':
        ufilename = dir_ + '1m_uids.txt'
        bfilename = dir_ + '1m_bids.txt'
        ufiles = ['1m_ratings_user.dat']
        vfiles = ['1m_ratings_item.dat']
    elif dt == 'ml-100k':
        ufilename = dir_ + '100k_uids.txt'
        bfilename = dir_ + '100k_bids.txt'
        ufiles = ['100k_ratings_user.dat']
        vfiles = ['100k_ratings_item.dat']
    elif dt == 'ml-10m':
        ufilename = dir_ + '10m_uids.txt'
        bfilename = dir_ + '10m_bids.txt'
        ufiles = ['10m_ratings_user.dat']
        vfiles = ['10m_ratings_item.dat']
    elif dt == 'amazon-app':
        ufilename = dir_ + 'filter5_uids.txt'
        bfilename = dir_ + 'filter5_bids.txt'
        ufiles = ['filter5_ratings_user.dat']
        vfiles = ['filter5_ratings_item.dat']
    elif dt == 'amazon-200k':
        ufilename = dir_ + 'uids.txt'
        bfilename = dir_ + 'bids.txt'
        ufiles = ['%s_top1000_user.dat' % path_str]
        vfiles = ['%s_top1000_item.dat' % path_str]

    if path_str in ['URNSRUB', 'URPSRUB', 'ratings_only']:
        ufiles = ['%s_user.dat' % path_str]
        vfiles = ['%s_item.dat' % path_str]


    logger.info('load user and item representations from %s and %s', ufiles, vfiles)
    uids = [int(l.strip()) for l in open(ufilename, 'r').readlines()]
    uid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in uids}

    bids = [int(l.strip()) for l in open(bfilename, 'r').readlines()]
    bid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in bids}

    for find, filename in enumerate(ufiles):
        ufs = np.loadtxt(dir_ + 'mf_features/path_count/' + filename, dtype=np.float64)
        cur = find * 10
        for uf in ufs:
            uid = int(uf[0])
            f = uf[1:]
            uid2reps[uid][cur:cur+10] = f

    for find, filename in enumerate(vfiles):
        bfs = np.loadtxt(dir_ + 'mf_features/path_count/' + filename, dtype=np.float64)
        cur = find * 10
        for bf in bfs:
            bid = int(bf[0])
            f = bf[1:]
            bid2reps[bid][cur:cur+10] = f
    return uid2reps, bid2reps

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

def run(lamb, K, eps, ite, solver, train_filename='', test_filename=''):
    '''
        基于block进行更新，同一个block内部的interaction为0
    '''
    L, F = 1, 10
    N = 2 * L * F
    eta = 1e-7#learning rate
    beta = 0.9#parameter used in line search
    run_start = time.time()

    b = 0.0
    W = np.random.rand(N) * 0.0001 # 1 by N
    V = np.random.rand(N, K) * 0.0001# N by K
    X, Y, test_X, test_Y = load_data(N, train_filename, test_filename)

    exp_info = 'exp on large scale data, normalized scale, op use lamb * eta, when reg=%s, solver=%s' % (lamb, solver)
    exp_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    logger.info('*************exp_time:%s,  %s*************', exp_time, exp_info)
    logger.info('fm_one_path started, %s group features, solver=%s, config K,reg,ite,eta,eps=(%s,%s,%s,%s,%s)', L, solver, K, lamb, ite, eta, eps)

    gw_inds = np.arange(N).reshape(2*L, F)
    gv_inds = np.arange(N*K).reshape(2*L, F*K)

    if solver == 'acc':
        rmses, maes = train_acc_prox_gradient(X, Y, W, V, b, gw_inds, gv_inds, test_X, test_Y, ite, eta, beta, eps, N, K)
    else:
        rmses, maes = train_prox_gradient(X, Y, W, V, b, gw_inds, gv_inds, test_X, test_Y, ite, eta, beta, eps, N, K)

    total_cost = (time.time() - run_start) / 60.0
    logger.info('fm_one_path finished, total_cost=%.2f mins exp_time=%s, %s group features, solver=%s, config K,reg,ite,eta, eps=(%s,%s,%s,%s,%s)', total_cost, exp_time, L, solver, K, lamb,ite,eta, eps)
    return rmses, maes

def train_prox_gradient(X, Y, W, V, b, gw_inds, gv_inds, test_X, test_Y, ite, eta, beta, eps, N, K):
    err = cal_err(X, Y, W, V, b)
    obj = np.power(err, 2).sum() + lamb * group_lasso(W, V.flatten(), gw_inds, gv_inds)
    objs = [obj]
    rmses = [cal_rmse(W, V, b, test_X, test_Y)]
    maes = [cal_mae(W, V, b, test_X, test_Y)]

    start = time.time()
    lt = 0
    ln = 1000
    for t in range(ite):
        start = time.time()
        #cal gradients
        #b = b - eta * 2 * err.sum()
        grad_w = 2 * np.dot(err, X)#element-wise correspondence
        part = np.dot(X, V)
        grad_v = np.zeros(V.shape)
        for i in range(N):
            tmp = np.square(X[:,i])
            for f in range(K):
                grad_v[i,f] = 2 * np.dot(err, np.multiply(X[:,i], part[:,f]) - V[i,f] * tmp)

        #line search with proximal operator
        if DEBUG:
            logger.debug('start line search...')
        for lt in range(ln+1):
            tW, tV = W - eta * grad_w, V - eta * grad_v
            W_p, V_p = prox_op(tW, tV, eta * lamb, gw_inds, gv_inds)
            err = cal_err(X, Y, W_p, V_p, b)
            obj_p = np.power(err, 2).sum() + lamb * group_lasso(W_p, V_p.flatten(), gw_inds, gv_inds)
            if DEBUG:
                logger.debug('lt=%s, obj_p=%s', lt, obj_p)
            if obj_p < objs[t]:
                objs.append(obj_p)
                W, V = W_p, V_p
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
        logger.info('exp_id=%s, iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', exp_id, t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
        if  dr < eps:
            break
    return rmses, maes

def cal_grad_v_by_c(err, X_i, part_f, tmp):
    xn = X_i.size
    nc = ctypes.c_int(xn)

    res = np.array([0.0], dtype=np.float64)
    resp = res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ep = err.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xp = X_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fp = part_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    tp = tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    cal_grad_v(ep, xp, fp, tp, nc, resp)
    return res[0]

def line_search(err, W, V, b, lamb, eta, gw_inds, gv_inds, obj_v, X, Y, ln, N, K):

    grad_start = time.time()
    grad_w = 2 * np.dot(err, X)#element-wise correspondence
    w_cost = time.time() - grad_start
    part = np.dot(X, V)
    grad_v = np.zeros(V.shape,dtype=np.float64)
    v_start = time.time()
    for i in range(N):
        tmp = np.square(X[:,i])
        for f in range(K):
            k_start = time.time()
            #grad_v[i,f] = cal_grad_v_by_c(err, X[:,i], part[:,f], V[i,f] * tmp)
            grad_v[i,f] = 2 * np.dot(err, np.multiply(X[:,i], part[:,f]) - V[i,f] * tmp)
    v_cost = time.time() - v_start

    #print 'grad_w/square/set/grads cost: %.2fs/%.2fs/%.2fs/%.2fs' % (w_cost, square_cost, v_set_cost, time.time() - grad_start)
    #line search with accelerated proximal operator
    for lt in range(ln+1):
        tW, tV = W - eta * grad_w, V - eta * grad_v
        W_p, V_p = prox_op(tW, tV, eta * lamb, gw_inds, gv_inds)
        l_err = cal_err(X, Y, W_p, V_p, b)
        l_obj = obj(l_err, W_p, V_p, lamb, gw_inds, gv_inds)
        if l_obj < obj_v:
            eta = 1.1 * eta
            break
        else:
            eta = 0.9 * eta
    return eta, lt, l_obj, W_p, V_p, v_cost

def train_acc_prox_gradient(X, Y, W, V, b, gw_inds, gv_inds, test_X, test_Y, ite, eta, beta, eps, N, K):
    '''
        accelerated proximal gradient method
    '''
    objs = [None] * (ite + 1)
    err = cal_err(X, Y, W, V, b)
    objs[0] = obj(err, W, V, lamb, gw_inds, gv_inds)
    rmses = [cal_rmse(W, V, b, test_X, test_Y)]
    maes = [cal_mae(W, V, b, test_X, test_Y)]

    A = np.hstack((W.reshape(-1,1), V))
    A0, A1, C1 = A.copy(), A.copy(), A.copy()
    c = objs[0]
    r0, r1, q, qeta = 0.0, 1.0, 1.0, 0.5
    eta1 = eta2 = eta

    lt1, lt2 = 0, 0
    ln = 1000
    for t in range(ite):
        start = time.time()
        #cal gradients
        B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
        W, V = B[:,0].flatten(), B[:,1:]

        err = cal_err(X, Y, W, V, b)
        obj_b = obj(err, W, V, lamb, gw_inds, gv_inds)

        b = b - eta * 2 * err.sum()

        l1start = time.time()
        eta1, lt1, obj_c, W_p, V_p, v_cost1 = line_search(err, W, V, b, lamb, eta1, gw_inds, gv_inds, obj_b, X, Y, ln, N, K)
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
            obj_a = obj(err, W, V, lamb, gw_inds, gv_inds)

            l2start = time.time()
            eta2, lt2, obj_v, W_p, V_p, v_cost2 = line_search(err, W, V, b, lamb, eta2, gw_inds, gv_inds, obj_a, X, Y, ln, N, K)
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
        logger.info('iter=%s, (lt1,eta1, v_cost1/cost)=(%s,%s, %.2f/%.2fs), (lt2,eta2,v_cost2/cost)=(%s,%s, %.2f/%.2fs), obj=%.5f(dr=%.8f), rmse=%.5f, mae=%.5f, cost=%.2f seconds', t, lt1, eta1, v_cost1, l1cost, lt2, eta2, v_cost2, l2cost, objs[t+1], dr, rmses[t+1], maes[t+1], (end - start))

        r0 = r1
        r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0
        tq = qeta * q + 1.0
        c = (qeta * q * c + objs[t+1]) / tq
        q = tq

        if  dr < eps:
            break

    np.savetxt('fm_res/W_%s_exp%s.txt' % (lamb, exp_id), W)
    np.savetxt('fm_res/V_%s_exp%s.txt' % (lamb, exp_id), V)
    return rmses, maes

def run_5_validation(lamb, K, eps, ite, solver):
    logger.info('start run_5_validations, path_str=%s, ratings_filename=%s, K=%s,eps=%s,reg=%s,iters=%s,solver=%s', path_str, rating_filename, K,eps,lamb, ite, solver)
    run_start = time.time()
    global dir_
    for rnd in xrange(5):
        start_time = time.time()
        dir_ = 'data/%s/exp_split/%s/' % (dt, rnd+1)
        train_filename = dir_ + '%s_train_%s.txt' % (rating_filename, rnd+1)
        test_filename = dir_ + '%s_test_%s.txt' % (rating_filename, rnd+1)
        logger.info('start validation %s, train_filename=%s, test_filename=%s', rnd+1, train_filename, test_filename)
        rmses, maes = run(lamb, K, eps, ite,solver, train_filename, test_filename)
        round_rmse = np.mean(rmses[-5:])
        round_mae = np.mean(maes[-5:])
        exp_rmses.append(round_rmse)
        exp_maes.append(round_mae)
        logger.info('finish validation %s, cost %.2f minutes, rmse=%.4f, mae=%.4f', rnd+1, (time.time() - start_time) / 60.0, exp_rmses[rnd], exp_maes[rnd])

    cost = (time.time() - run_start) / 60.0
    logger.info('**********finish run_5_validations, path_str=%s, cost %.2f mins, rating_filename=%s***********\n*****config: (reg, K, eps, iters solver)=(%s, %s, %s, %s, %s), exp rmses: %s, maes: %s\n*******avg rmse=%s, avg mae=%s\n**************', path_str, cost, rating_filename, lamb, K, eps, ite, solver, exp_rmses, exp_maes, np.mean(exp_rmses), np.mean(exp_maes))

if __name__ == '__main__':
    if len(sys.argv) == 4:
        global dir_
        global path_str
        path_str = sys.argv[3]
        init_conifg(sys.argv[2])
        if int(sys.argv[1]) == 1:
            rnd = 1
            lamb = 0
            ite = 10000
            K, solver = 100, 'acc'
            eps = 1e-8
            run_start = time.time()
            dir_ = 'data/%s/exp_split/%s/' % (dt, rnd+1)
            train_filename = dir_ + '%s_train_%s.txt' % (rating_filename, rnd+1)
            test_filename = dir_ + '%s_test_%s.txt' % (rating_filename, rnd+1)
            rmses, maes = run(lamb, K, eps, ite, solver, train_filename, test_filename)
            cost = (time.time() - run_start) / 60.0
            logger.info('**********fm_one_path for %s finish, run once, cost %.2f mins*******\nconfig: (reg, K, eps, ites, solver)=(%s, %s, %s, %s, %s), rmses: %s, maes: %s\navg rmse=%s, avg mae=%s\n***************', path_str, cost, lamb, K, eps, ite, solver, rmses[-5:], maes[-5:], np.mean(rmses[-5:]), np.mean(maes[-5:]))
        elif int(sys.argv[1]) == 2:
            exp_rmses, exp_maes = [], []
            lamb, K, ite, solver = 1, 10, 500, 'acc'
            eps = 1e-8
            run_5_validation(lamb, K, eps, ite, solver)
    else:
        print 'please specify dataset: yelp, ml-100k,ml-1m,ml-10m, amazon-200k'
        sys.exit(0)
