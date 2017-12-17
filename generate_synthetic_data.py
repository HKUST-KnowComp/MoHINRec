#coding=utf8
'''
    generate synthetic data for testing group lasso
'''
import os
import numpy as np

dir_ = 'data/random_first_order/'

if not os.path.isdir(dir_):
    os.makedirs(dir_)

L = 10
F = 10
N = 2 * L * F
M = 10000
K = 10

def run():
    X = np.random.rand(M, N)
    W = np.random.rand(N)
    P = np.random.rand(N, K)
    gw_inds = np.arange(N).reshape(2 * L, F)
    gp_inds = np.arange(N * K).reshape(2 * L, F * K)
    inds = np.random.permutation(np.arange(2 * L))
    W[gw_inds[inds[:10]]] = 0.0
    inds = np.random.permutation(np.arange(2 * L))
    P[gw_inds[inds[:10]]] = 0.0
    WX = np.dot(W, X.T)
    XP = np.dot(X, P)
    XSPS = np.dot(np.square(X), np.square(P))
    #Y = WX + 0.5 * (np.square(XP) - XSPS).sum(axis=1)
    Y = WX
    #Y = 0.5 * (np.square(XP) - XSPS).sum(axis=1)
    #Y += np.random.rand(M) #noise
    train_X, train_Y = X[:8000], Y[:8000]
    test_X, test_Y = X[8000:], Y[8000:]
    np.savetxt(dir_ + 'train_X.txt', train_X)
    np.savetxt(dir_ + 'train_Y.txt', train_Y)
    np.savetxt(dir_ + 'test_X.txt', test_X)
    np.savetxt(dir_ + 'test_Y.txt', test_Y)
    np.savetxt(dir_ + 'W.txt', W)
    np.savetxt(dir_ + 'P.txt', P)

if __name__ == '__main__':
    run()
