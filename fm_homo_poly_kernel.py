#coding=utf8
'''
    Implement the Factorization Machine by the homogeneous polynomial kernel
'''
import time
import logging

import numpy as np
from numpy.linalg import norm

from exp_util import cal_rmse, cal_mae

class FMHPK(object):

    def __init__(self, config, data_loader):
        self.config = config
        #self.logger = self.config.get('logger')
        self.train_X, self.train_Y, self.test_X, self.test_Y = data_loader.get_exp_data()
        self._init_config()

    def _init_config(self):
        self.exp_id = self.config.get('exp_id')
        self.N = self.config.get('N')
        self.K = self.config.get('K')
        self.initial = self.config.get('initial')
        self.bias = self.config.get('bias')
        self.reg_W = self.config.get('reg_W')
        self.reg_P = self.config.get('reg_P')
        self.reg_Q = self.config.get('reg_Q')
        self.max_iters = self.config.get('max_iters')
        self.ln = self.config.get('ln')
        self.eps = self.config.get('eps')
        self.eta = self.config.get('eta')
        #better to add log information for the configs
        self.M = self.train_X.shape[0]

    def _obj(self, err, W, P, Q):
        return np.power(err, 2).sum() / self.M + self.reg_W * np.power(norm(W), 2) + self.reg_P * np.power(norm(P), 2) + self.reg_Q * np.power(norm(Q), 2)

    def _cal_err(self, WX, XP, XQ, Y):
        part = 0.5 * np.multiply(XP, XQ)
        Y_t = self.bias + WX + part.sum(axis=1)
        return Y_t - Y

    def _get_XC_prods(self, X, W, P, Q):
        WX = np.dot(W, X.T)
        XP = np.dot(X, P)
        XQ = np.dot(X, Q)
        return WX, XP, XQ

    def get_eval_res(self):
        return self.rmses, self.maes

    def train(self):

        W = np.random.rand(self.N) * self.initial # 1 by N
        P = np.random.rand(self.N, self.K) * self.initial# N by K
        Q = np.random.rand(self.N, self.K) * self.initial# N by K

        self._block_gradient_descent(W, P, Q)

    def _block_gradient_descent(self, W, P, Q):
        WX, XP, XQ = self._get_XC_prods(self.train_X, W, P, Q)
        err = self._cal_err(WX, XP, XQ, self.train_Y)
        objs = [self._obj(err, W, P, Q)]

        WtX, tXP, tXQ = self._get_XC_prods(self.test_X, W, P, Q)
        test_err = self._cal_err(WtX, tXP, tXQ, self.test_Y)
        rmses = [cal_rmse(test_err)]
        maes = [cal_mae(test_err)]

        start = time.time()

        eta = self.eta
        for t in range(self.max_iters):
            start = time.time()
            #cal gradients
            WX, XP, XQ = self._get_XC_prods(self.train_X, W, P, Q)
            err = self._cal_err(WX, XP, XQ, self.train_Y)
            grad_W = 2.0 / self.M * np.dot(err, self.train_X)#element-wise correspondence
            grad_W += 2 * self.reg_W * W

            grad_P = np.zeros(P.shape)
            for f in range(self.K):
                grad_P[:,f] = np.dot(err, np.multiply(self.train_X, XQ[:,f].reshape(-1,1).repeat(self.N, axis=1)))
            grad_P /= self.M
            grad_P += 2 * self.reg_P * P

            grad_Q = np.zeros(Q.shape)
            for f in range(self.K):
                grad_Q[:,f] = np.dot(err, np.multiply(self.train_X, XP[:,f].reshape(-1,1).repeat(self.N, axis=1)))
            grad_Q /= self.M
            grad_Q += 2 * self.reg_Q * Q

            #import pdb;pdb.set_trace()
            l_obj, eta, lt, W, P, Q = self._line_search(objs[t], eta, W, P, Q, grad_W, grad_P, grad_Q)

            if lt == self.ln:
                logging.info('!!!stopped by line_search, lt=%s!!!', lt)
                break

            objs.append(l_obj)

            WtX, tXP, tXQ = self._get_XC_prods(self.test_X, W, P, Q)
            test_err = self._cal_err(WtX, tXP, tXQ, self.test_Y)
            rmses.append(cal_rmse(test_err))
            maes.append(cal_mae(test_err))
            end = time.time()

            dr = abs(objs[t] - objs[t+1]) / objs[t]
            logging.info('exp_id=%s, iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', self.exp_id, t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
            if  dr < self.eps:
                logging.info('*************stopping criterion satisfied*********')
                break

        self.rmses, self.maes = rmses, maes

    def _line_search(self, obj_v, eta, W, P, Q, grad_W, grad_P, grad_Q):
        for lt in range(self.ln+1):
            lW, lP, lQ = W - eta * grad_W, P - eta * grad_P, Q - eta * grad_Q
            lWX, XlP, XlQ = self._get_XC_prods(self.train_X, lW, lP, lQ)
            l_err = self._cal_err(lWX, XlP, XlQ, self.train_Y)
            l_obj = self._obj(l_err, lW, lP, lQ)
            if l_obj < obj_v:
                eta = 1.1 * eta
                W, P, Q = lW, lP, lQ
                break
            else:
                eta = 0.7 * eta
        return l_obj, eta, lt, W, P, Q

    def _save_paras(self, W, P, Q):
        pass
        #split_num = self.data_dir.split('/')[-2]
        #W_wfilename = 'fm_res/split%s_W_%s_exp%s.txt' % (split_num, lambw, exp_id)
        #np.savetxt(W_wfilename, W)
        #V_wfilename = 'fm_res/split%s_V_%s_exp%s.txt' % (split_num, lambv, exp_id)
        #np.savetxt(V_wfilename, V)
        #self.logger.info('W and V saved in %s and %s', W_wfilename, V_wfilename)
