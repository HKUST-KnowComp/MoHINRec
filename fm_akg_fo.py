#coding=utf8
'''
    standard fm which only captures the first order interactions
'''
import time
import logging

import numpy as np
from numpy.linalg import norm

from exp_util import cal_rmse, cal_mae

from fm_anova_kernel_glasso import FMAKGL

class FMAKGL_FO(FMAKGL):

    def _obj(self, err, W):
        return np.power(err, 2).sum() + self.reg_W * self._group_lasso(W, self.gw_inds)

    def _cal_err(self, WX, Y):
        Y_t = self.bias + WX
        return Y_t - Y

    def _get_XC_prods(self, X, W):
        WX = np.dot(W, X.T)
        return WX

    def get_eval_res(self):
        return self.rmses, self.maes

    def train(self):

        if self.config.get('dt') == 'synthetic':
            dir_ = self.config.get('data_dir')
            ground_W = np.loadtxt(dir_ + 'W.txt')
            W = ground_W + np.random.rand(self.N) * self.initial # N by K
            print 'synthetic W'
        else:
            W = np.random.rand(self.N) * self.initial # 1 by N
        self.gw_inds = np.arange(self.N).reshape(2 * self.L, self.F)

        if self.solver == 'PG':
            self._block_proximal_gradient_descent(W)
        elif self.solver == 'mAPG':
            self._block_mono_acc_proximal_gradient_descent(W)

    def _block_proximal_gradient_descent(self, W):
        WX = self._get_XC_prods(self.train_X, W)
        err = self._cal_err(WX, self.train_Y)
        objs = [self._obj(err, W)]

        WtX = self._get_XC_prods(self.test_X, W)
        test_err = self._cal_err(WtX, self.test_Y)
        rmses = [cal_rmse(test_err)]
        maes = [cal_mae(test_err)]

        start = time.time()

        eta = self.eta
        for t in range(self.max_iters):
            start = time.time()

            l_obj, eta, lt, W = self._get_updated_paras(objs[t], eta, W)

            if lt == self.ln:
                logging.info('!!!stopped by line_search, lt=%s!!!', lt)
                break

            objs.append(l_obj)

            WtX = self._get_XC_prods(self.test_X, W)
            test_err = self._cal_err(WtX, self.test_Y)
            rmses.append(cal_rmse(test_err))
            maes.append(cal_mae(test_err))
            end = time.time()

            dr = abs(objs[t] - objs[t+1]) / objs[t]
            logging.info('exp_id=%s, iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', self.exp_id, t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
            if  dr < self.eps:
                logging.info('*************stopping criterion satisfied*********')
                break

        logging.info('train process finished, total iters=%s', t+1)
        self.rmses, self.maes = rmses, maes
        self._save_paras(W)

    def _block_mono_acc_proximal_gradient_descent(self, W, P):
        '''
            monotone accelerated pg
        '''

        logging.info('start solving by _block_mono_acc_proximal_gradient_descent')
        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        objs = [self._obj(err, W, P)]

        WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
        test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
        rmses = [cal_rmse(test_err)]
        maes = [cal_mae(test_err)]

        start = time.time()

        A = np.hstack((W.reshape(-1,1), P))
        A0, A1, C1 = A.copy(), A.copy(), A.copy()
        r0, r1 = 0, 1

        eta = self.eta
        XS = np.square(self.train_X)
        for t in range(self.max_iters):
            start = time.time()
            #cal gradients
            v_obj, v_eta, v_lt, vW, vP = self._get_updated_paras(objs[t], eta, W, P)

            B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
            W, P = B[:,0].flatten(), B[:,1:]
            y_obj, y_eta, y_lt, yW, yP = self._get_updated_paras(objs[t], eta, W, P)
            C1 = np.hstack((yW.reshape(-1,1), yP))

            if v_obj > y_obj:
                objs.append(y_obj)
                lt = y_lt
                eta = y_eta
                W, P = yW, yP
            else:
                objs.append(v_obj)
                lt = v_lt
                eta = v_eta
                W, P = vW, vP

            if lt == self.ln:
                logging.info('!!!stopped by line_search, lt=%s!!!', lt)
                break

            A0 = A1
            A1 = np.hstack((W.reshape(-1,1), P))
            r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0

            WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
            test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
            rmses.append(cal_rmse(test_err))
            maes.append(cal_mae(test_err))
            end = time.time()

            dr = abs(objs[t] - objs[t+1]) / objs[t]
            logging.info('exp_id=%s, iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', self.exp_id, t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
            if  dr < self.eps:
                logging.info('*************stopping criterion satisfied*********')
                break

        logging.info('train process finished, total iters=%s', t+1)
        self.rmses, self.maes = rmses, maes
        self._save_paras(W)

    def _get_updated_paras(self, obj_t, eta, W):

        WX = self._get_XC_prods(self.train_X, W)
        err = self._cal_err(WX, self.train_Y)

        #cal gradients
        grad_W = 2 * np.dot(err, self.train_X)#element-wise correspondence

        l_obj, eta, lt, W = self._line_search(obj_t, eta, W, grad_W)

        return l_obj, eta, lt, W

    def _line_search(self, obj_v, eta, W, grad_W):
        for lt in range(self.ln+1):
            lW = W - eta * grad_W
            lW = self._prox_op(eta * self.reg_W, lW, self.gw_inds)

            lWX = self._get_XC_prods(self.train_X, lW)
            l_err = self._cal_err(lWX, self.train_Y)
            l_obj = self._obj(l_err, lW)

            if l_obj < obj_v:
                eta *= 1.1
                W = lW
                break
            else:
                eta *= 0.7
        return l_obj, eta, lt, W

    def _save_paras(self, W):
        split_num = self.config.get('data_dir').split('/')[-2]
        dt = self.config.get('dt')
        W_wfilename = 'fm_res/%s_split%s_W_%s_exp%s.txt' % (dt, split_num, self.reg_W, self.exp_id)
        np.savetxt(W_wfilename, W)
        logging.info('W saved in %s', W_wfilename)



