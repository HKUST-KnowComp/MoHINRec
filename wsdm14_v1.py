#coding=utf8
'''
    codes for implementing the recommendation model in wsdm14
    the basic one, cluster mode is not implemented right now
    20170207: add acc solver
'''
import sys
import time
from datetime import datetime
import logging

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from numba import jit

from logging_util import init_logger

INCLUDE_RATINGS = True

def init_conifg(dt_arg, reg, exp_type):
    global rating_filename
    global logger
    global exp_id
    global dt
    dt = dt_arg

    if dt == 'yelp':
        dir_ = 'data/yelp/'
        rating_filename = 'ratings_filter5'
    elif dt == 'yelp-200k':
        rating_filename = 'ratings'
    elif dt == 'cikm-yelp':
        rating_filename = 'ratings'
    elif dt == 'yelp-sample':
        dir_ = 'data/yelp/samples/'
        rating_filename = ''
    elif dt in ['ml-100k', 'ml-1m', 'ml-10m']:
        dir_ = 'data/movielens/'
        rating_filename = '%s-rating' % dt
    elif dt == 'amazon-app':
        dir_ = 'data/amazon/'
        rating_filename = 'filter5_ratings_Apps_for_Android'
    elif dt == 'amazon':
        dir_ = 'data/amazon/'
        rating_filename = 'ratings_filter5'

    log_filename = 'log/%s_wsdm_v1_reg%s.log' % (dt, reg)
    if exp_type == 1:
        log_filename = 'log/%s_wsdm_v1_once_reg%s.log' % (dt, reg)
    exp_id = int(time.time())
    logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)

def load_representation():
    '''
        load all the latent features from files, which are generated based on meta-path similarity
    '''
    uid2reps = {}
    bid2reps = {}

    #ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBStarsB_top1000_user.dat', 'UPBStateB_top1000_user.dat', 'UPBCityB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'UUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
    #vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBStarsB_top1000_item.dat', 'UPBStateB_top1000_item.dat', 'UPBCityB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'UUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']
    ufiles = ['URPSRUB_user.dat', 'UPBCatB_top1000_user.dat',]
    vfiles = ['URPSRUB_item.dat', 'UPBCatB_top1000_item.dat',]
    if dt == 'cikm-yelp':
        ufiles = ['UPBCatBUB_top500_user.dat', 'UPBCityBUB_top500_user.dat']
        vfiles = ['UPBCatBUB_top500_item.dat', 'UPBCityBUB_top500_item.dat']

    if INCLUDE_RATINGS:
        ufiles.append('ratings_only_user.dat')
        vfiles.append('ratings_only_item.dat')

    for find, filename in enumerate(ufiles):
        ufs = np.loadtxt(dir_ + 'mf_features/path_count/' + filename)
        for uf in ufs:
            uid = int(uf[0])
            f = uf[1:]
            uid2reps.setdefault(uid, {})[find] = f

    for find, filename in enumerate(vfiles):
        bfs = np.loadtxt(dir_ + 'mf_features/path_count/' + filename)
        for bf in bfs:
            bid = int(bf[0])
            f = bf[1:]
            bid2reps.setdefault(bid, {})[find] = f
    return uid2reps, bid2reps

def load_data(train_filename, test_filename):
    train_data = np.loadtxt(train_filename)
    test_data = np.loadtxt(test_filename)
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]
    logger.info('train_data: (%.4f,%.4f), test_data: (%.4f,%.4f)', np.mean(train_data[:,2]), np.std(train_data[:,2]), np.mean(test_data[:,2]), np.std(test_data[:,2]))
    return train_data, test_data

#@jit(arg_types=[double[:], double[:,:], double[:,:], double[:,:]])
def cal_omega(thetas, Us, Vs, train_data):
    sum_err = np.zeros(train_data.shape[0])
    sum_err = []
    for u, b, r in train_data:
        pr = 0.0
        for i, theta in enumerate(thetas):
            if i in Us.get(u, {}).keys() and i in Vs.get(b,{}).keys():
                pr += theta * np.dot(Us[u][i], Vs[b][i])
        #sum_err += np.power((pr - r), 2)
        sum_err.append(pr - r)
    return np.array(sum_err)

def obj(thetas, Us, Vs, train_data, lamb):
    sum_err = 0.0
    for u, b, r in train_data:
        pr = 0.0
        for i, theta in enumerate(thetas):
            if i in Us.get(u, {}).keys() and i in Vs.get(b,{}).keys():
                pr += theta * np.dot(Us[u][i], Vs[b][i])
        sum_err += np.power((pr - r), 2)
    sum_err += lamb * np.power(norm(thetas),2)
    return sum_err

def cal_grad(thetas, Us, Vs, train_data, lamb):
    grad_ts = np.zeros(len(thetas))
    for u, b, r in train_data:
        pr = 0.0
        for i, theta in enumerate(thetas):
            if i in Us.get(u, {}).keys() and i in Vs.get(b,{}).keys():
                pr += theta * np.dot(Us[u][i], Vs[b][i])
        err = pr - r
        for i in range(len(thetas)):
            if i in Us.get(u, {}).keys() and i in Vs.get(b,{}).keys():
                grad_ts[i] += err * np.dot(Us[u][i], Vs[b][i])
    grad_ts += lamb * thetas
    grad_ts *= 2
    return grad_ts

def cal_measures(thetas, Us, Vs, data):
    rmse_err = 0.0
    mae_err = 0.0
    num = data.shape[0]
    for u, b, r in data:
        pr = 0.0
        for i, theta in enumerate(thetas):
            if i in Us.get(u, {}).keys() and i in Vs.get(b,{}).keys():
                pr += theta * np.dot(Us[u][i], Vs[b][i])
        rmse_err += np.power((pr - r), 2)
        mae_err += abs(pr - r)
    rmse = np.sqrt(rmse_err / num)
    mae = mae_err / num
    return rmse, mae

def run(lamb, eps, ite, train_filename, test_filename):
    L = 2
    if INCLUDE_RATINGS:
        L += 1
    thetas = np.random.rand(L) * 0.000001
    Us, Vs = load_representation()
    train_data, test_data = load_data(train_filename, test_filename)
    eta, ln = 1e-7, 100
    solver = 'acc'

    exp_id = int(time.time())
    exp_info = 'wsdm14 v1 on %s, config: L=%s, reg=%s' % (dt, L,lamb)
    exp_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    logger.info('*************exp_time:%s, exp_id=%s, %s*************', exp_time, exp_id, exp_info)
    logger.info('wsdm14 v1 started, exp_id=%s, exp_time=%s, L=%s, reg=%s, thetas=%s', exp_id, exp_time, L, lamb, thetas)
    if solver == 'acc':
        rmses, maes = train_acc(thetas, lamb, eta, eps, ite, ln, Us, Vs, train_data, test_data)
    logger.info('wsdm14 v1 finished, exp_id=%s, exp_time=%s, L=%s, reg=%s, thetas=%s', exp_id, exp_time, L, lamb, thetas)
    return rmses, maes

def line_search(obj_v, thetas, grad_ts, eta, lamb, ln, Us, Vs, train_data):
    for ls in range(ln+1):
        l_thetas = thetas - eta * grad_ts
        lobj = obj(l_thetas, Us, Vs, train_data, lamb)
        if lobj < obj_v:
            eta *= 1.1
            break
        else:
            eta *= 0.7
    return ls, eta, lobj, l_thetas

def train(thetas, lamb, eta, eps, ite, ln, Us, Vs, train_data, test_data):
    objs = [None] * (ite + 1)
    rmse, mae = cal_measures(thetas, Us, Vs, test_data)
    rmses, maes = [rmse], [mae]
    for t in range(ite):
        start = time.time()
        grad_ts = cal_grad(thetas, Us, Vs, train_data, lamb)

        for ls in range(ln):
            tmp_thetas = thetas - eta * grad_ts
            lobj = obj(tmp_thetas, Us, Vs, train_data, lamb)
            if lobj < objs[t]:
                objs.append(lobj)
                thetas = tmp_thetas
                eta *= 1.1
                break
            else:
                eta *= 0.7

        rmse, mae = cal_measures(thetas, Us, Vs, test_data)
        rmses.append(rmse)
        maes.append(mae)

        lrate = (objs[t] - objs[t+1]) / objs[t]
        if abs(lrate) < eps:
            break

        if ls + 1 == ln:
            break

        end = time.time()
        logger.info('iter=%s, obj=%.4f(%.7f%%), ls:((%.4f, %s), rmse=%.4f, mae=%.4f, cost: %.2fs', t+1, objs[t+1], lrate * 100, eta, ls, rmses[t+1], maes[t+1], end-start)
    return rmses, maes

def train_acc(thetas, lamb, eta, eps, ite, ln, Us, Vs, train_data, test_data):
    objs = [None] * (ite + 1)
    rmse, mae = cal_measures(thetas, Us, Vs, test_data)
    rmses, maes = [rmse], [mae]
    objs[0] = obj(thetas, Us, Vs, train_data, lamb)

    A0, A1, C1 = thetas.copy(), thetas.copy(), thetas.copy()
    c = objs[0]
    r0, r1, q, qeta = 0.0, 1.0, 1.0, 0.5

    for t in range(ite):
        start = time.time()

        B = A1 + r0 * (C1 - A1) / r1 + (r0 - 1)/r1 * (A1 - A0)
        obj_b = obj(B, Us, Vs, train_data, lamb)
        grad_b = cal_grad(B, Us, Vs, train_data, lamb)

        ls1, eta, obj_c, lc = line_search(obj_b, B, grad_b, eta, lamb, ln, Us, Vs, train_data)

        if ls1 == ln:
            logger.info('ls1=%s, terminate iterating!!!', ln)
            break

        A0 = A1.copy()
        ls2 = 0
        if obj_c < c:
            A1 = lc.copy()
            objs[t+1] = obj_c
        else:
            obj_a = obj(A1, Us, Vs, train_data, lamb)
            grad_a = cal_grad(A1, Us, Vs, train_data, lamb)
            ls2, eta, obj_d, ld = line_search(obj_a, A1, grad_a, eta, lamb, ln, Us, Vs, train_data)

            if ls2 == ln:
                logger.info('ls2=%s, terminate iterating!!!', ln)
                break

            if obj_c < obj_d:
                A1 = lc.copy()
                objs[t+1] = obj_c
            else:
                A1 = ld.copy()
                objs[t+1] = obj_d

        thetas = A1.copy()
        rmse, mae = cal_measures(thetas, Us, Vs, test_data)
        rmses.append(rmse)
        maes.append(mae)

        r0 = r1
        r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0
        tq = qeta * q + 1.0
        c = (qeta * q * c + objs[t+1]) / tq
        q = tq

        lrate = (objs[t] - objs[t+1]) / objs[t]

        if abs(lrate) < eps:
            break

        end = time.time()
        logger.info('iter=%s, obj=%.4f(%.7f%%), eta:%s, ls:((%s, %s), rmse=%.4f, mae=%.4f, cost: %.2fs', t+1, objs[t+1], lrate * 100, eta, ls1, ls2, rmses[t+1], maes[t+1], end-start)
    return rmses, maes

def plot(objs, rmses):
    X = np.arange(len(objs))
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(X, objs)
    axarr[0].set_title('objs and test rmses')
    axarr[1].plot(X, rmses)
    plt.show()

def run_5_validation(lamb, eps, ite):
    logger.info('start run_5_validations, ratings_filename=%s, eps=%s,reg=%s,iters=%s', rating_filename, eps, lamb, ite)
    run_start = time.time()
    exp_rmses, exp_maes = [], []
    global dir_
    for rnd in xrange(5):
        start_time = time.time()
        dir_ = 'data/yelp-200k/exp_split/%s/' % (rnd+1)
        train_filename = dir_ + '%s_train_%s.txt' % (rating_filename, rnd+1)
        test_filename = dir_ + '%s_test_%s.txt' % (rating_filename, rnd+1)
        logger.info('start validation %s, train_filename=%s, test_filename=%s', rnd+1, train_filename, test_filename)

        rmses, maes = run(lamb, eps, ite, train_filename, test_filename)

        round_rmse = np.mean(rmses[-5:])
        round_mae = np.mean(maes[-5:])
        exp_rmses.append(round_rmse)
        exp_maes.append(round_mae)
        logger.info('finish validation %s, cost %.2f minutes, rmse=%.4f, mae=%.4f', rnd+1, (time.time() - start_time) / 60.0, exp_rmses[rnd], exp_maes[rnd])

    cost = (time.time() - run_start) / 60.0
    logger.info('**********finish run_5_validations, cost %.2f mins, rating_filename=%s***********\n*****config: (reg, eps, iters)=(%s, %s, %s), exp rmses: %s, maes: %s\n*******avg rmse=%s, avg mae=%s\n**************', cost, rating_filename, lamb, eps, ite, exp_rmses, exp_maes, np.mean(exp_rmses), np.mean(exp_maes))

if __name__ == '__main__':
    if len(sys.argv) == 4:
        dt = sys.argv[1]
        lamb = float(sys.argv[3].replace('reg',''))
        exp_type = int(sys.argv[2])
        eps = 1e-7
        ite = 200
        init_conifg(dt, lamb, exp_type)
        global dir_

        if exp_type == 1:
            split_num = 2
            dir_ = 'data/%s/exp_split/%s/' % (dt, split_num)
            train_filename = dir_ + '%s_train_%s.txt' % (rating_filename, split_num)
            test_filename = dir_ + '%s_test_%s.txt' % (rating_filename, split_num)
            logger.info('start validation %s, train_filename=%s, test_filename=%s', split_num, train_filename, test_filename)
            run_start = time.time()
            rmses, maes = run(lamb, eps, ite, train_filename, test_filename)
            cost = time.time() - run_start
            logger.info('finish validation %s, cost %.2fs, lamb,eps,ite=(%s,%s,%s), train_filename=%s, test_filename=%s', split_num, cost, lamb, eps, ite, train_filename, test_filename)
            logger.info('rmse=%s, mae=%s', np.mean(rmses[-5:]), np.mean(maes[-5:]))
        elif int(sys.argv[2]) == 2:
            run_5_validation(lamb, eps, ite)
