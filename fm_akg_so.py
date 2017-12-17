#coding=utf8
'''
    standard fm which only captures the second order interactions
'''
import time
import logging

import numpy as np
from numpy.linalg import norm

from exp_util import cal_rmse, cal_mae

from fm_anova_kernel_glasso import FMAKGL

class FMAKGL_SO(FMAKGL):

    def _obj(self, err, P):
        part1 = np.power(err, 2).sum() / self.M
        part2 = self.reg_P * self._group_lasso(P.flatten(), self.gp_inds)
        logging.info('obj detail, part1=%s, part2=%s', part1, part2)
        return part1 + part2

    def _cal_err(self, XP, XSPS, Y):
        Y_t = self.bias + 0.5 * (np.square(XP) - XSPS).sum(axis=1)
        return Y_t - Y

    def _get_XC_prods(self, X, P):
        XP = np.dot(X, P)
        XSPS = np.dot(np.square(X), np.square(P))
        return XP, XSPS

    def get_eval_res(self):
        return self.rmses, self.maes

    def train(self):

        if self.config.get('dt') == 'synthetic':
            dir_ = self.config.get('data_dir')
            ground_P = np.loadtxt(dir_ + 'P.txt')
            P = ground_P + np.random.rand(self.N, self.K) * self.initial # N by K
            print 'synthetic P'
        else:
            P = np.random.rand(self.N, self.K) * self.initial# N by K
        self.gp_inds = np.arange(self.N * self.K).reshape(2 * self.L, self.F * self.K)

        if self.solver == 'PG':
            self._block_proximal_gradient_descent(P)
        elif self.solver == 'mAPG':
            self._block_mono_acc_proximal_gradient_descent(P)

    def _block_proximal_gradient_descent(self, P):
        XP, XSPS = self._get_XC_prods(self.train_X, P)
        err = self._cal_err(XP, XSPS, self.train_Y)
        objs = [self._obj(err, P)]

        tXP, tXSPS = self._get_XC_prods(self.test_X, P)
        test_err = self._cal_err(tXP, tXSPS, self.test_Y)
        rmses = [cal_rmse(test_err)]
        maes = [cal_mae(test_err)]

        start = time.time()

        eta = self.eta
        for t in range(self.max_iters):
            start = time.time()

            l_obj, eta, lt, P = self._get_updated_paras(objs[t], eta, P)

            if lt == self.ln:
                logging.info('!!!stopped by line_search, lt=%s!!!', lt)
                break

            objs.append(l_obj)

            tXP, tXSPS = self._get_XC_prods(self.test_X, P)
            test_err = self._cal_err(tXP, tXSPS, self.test_Y)
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
        self._save_paras(P)

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
            v_obj, v_eta, v_lt, vW, vP = self._get_updated_paras(objs[t], eta, P)

            B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
            W, P = B[:,0].flatten(), B[:,1:]
            y_obj, y_eta, y_lt, yW, yP = self._get_updated_paras(objs[t], eta, P)
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
        self._save_paras(P)

    def _get_updated_paras(self, obj_t, eta, P):

        XP, XSPS = self._get_XC_prods(self.train_X, P)
        err = self._cal_err(XP, XSPS, self.train_Y)

        XS = np.square(self.train_X)
        grad_P = np.zeros(P.shape)
        for f in range(self.K):
            grad_P[:,f] = 2.0 / self.M * np.dot(err, np.multiply(self.train_X, XP[:,f].reshape(-1,1).repeat(self.N, axis=1)) - np.multiply(P[:,f].reshape(1, -1).repeat(self.M, axis=0), XS))

        l_obj, eta, lt, P = self._line_search(obj_t, eta, P, grad_P)

        return l_obj, eta, lt, P

    def _line_search(self, obj_v, eta, P, grad_P):
        for lt in range(self.ln+1):
            lP = P - eta * grad_P
            lP = self._prox_op(eta * self.reg_P, lP.flatten(), self.gp_inds)
            lP = lP.reshape(P.shape)

            XlP, XSlPS = self._get_XC_prods(self.train_X, lP)
            l_err = self._cal_err(XlP, XSlPS, self.train_Y)
            l_obj = self._obj(l_err, lP)

            if l_obj < obj_v:
                eta *= 1.2
                P = lP
                break
            else:
                eta *= 0.8
        return l_obj, eta, lt, P

    def _save_paras(self, P):
        split_num = self.config.get('data_dir').split('/')[-2]
        dt = self.config.get('dt')
        P_wfilename = 'fm_res/%s_split%s_P_%s_exp%s.txt' % (dt, split_num, self.reg_P, self.exp_id)
        np.savetxt(P_wfilename, P)
        logging.info('P saved in %s', P_wfilename)

