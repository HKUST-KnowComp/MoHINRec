#coding=utf8
'''
    linear regression with group lasso, solved by proximal gradient methods
'''

import numpy as np

from numpy.linalg import norm
from numpy import power
from sklearn import datasets, linear_model


import matplotlib.pyplot as plt

stf = lambda b: b if b > 0.0 else 0.0#soft threshold function
lamb = 100

def prox_op(W, eta, gw_inds):
    for i in range(len(gw_inds)):
        W[gw_inds[i]] = stf(1 - eta / norm(W[gw_inds[i]])) * W[gw_inds[i]]
    return W

def group_lasso(W, gw_inds):
    #import pdb;pdb.set_trace()
    res = 0.0
    for i in range(len(gw_inds)):
        res += norm(W[gw_inds[i]])
    return res

def obj(X, W, Y, gw_inds, fnorm=False):
    if fnorm:
        return 0.5 * power(norm(np.dot(X, W) - Y),2)  + lamb * power(norm(W), 2)
    else:
        return 0.5 * power(norm(np.dot(X, W) - Y),2)  + lamb * group_lasso(W, gw_inds)

def rmse(test_X, W, test_Y):
    err = np.dot(test_X, W) - test_Y
    num = test_Y.size
    return np.sqrt(np.square(err).sum() / num)

def line_search(obj_v, W, grad_w, X, Y, ln, eta, lamb, gw_inds):
    lW = W - eta * grad_w
    lW = prox_op(lW, eta * lamb, gw_inds)
    lobj = obj(X, lW, Y, gw_inds)

    for ls in range(ln+1):
        if lobj < obj_v:
            eta *= 1.1
            break
        else:
            eta *= 0.7
    return eta, ls, lobj, lW

def fit_lr(X, Y, test_X, test_Y):
    N, M = X.shape
    W = np.random.rand(M) * 1e-7
    #gw_inds = np.array([[0,1,2,3],[4,5],[6,7,8,9],[10,]])
    gw_inds = np.array([[0,1,2],[3,4], [5, 7], [6], [8,9]])
    eta = 1e-7
    ls = 0
    max_iter = 1000
    ln = 200
    objs = [obj(X, W, Y, gw_inds)]
    rmses = [rmse(test_X, W, test_Y)]
    for it in range(max_iter):
        print 'iter=%s, obj=%s, test rmse=%s, ls,eta=%s,%s' % (it, objs[it], rmses[it], ls,eta)
        #import pdb;pdb.set_trace()
        tmp = np.dot(X, W) - Y
        grad_w = np.zeros(M)
        for j in range(M):
            grad_w[j] = np.dot(tmp, X[:,j])
        #grad_w = grad_w + 2 * lamb * W

        eta, ls, lobj, lW = line_search(objs[it], W, grad_w, X, Y, ln, eta, lamb, gw_inds)
        if ls == ln:
            print ls
            break
        W = lW
        objs.append(obj(X, W, Y, gw_inds))
        rmses.append(rmse(test_X, W, test_Y))
        dr = abs((objs[it+1] - objs[it]) / objs[it])
        if dr < 1e-8:
            break
    print 'finish,iter=%s, rmse=%s test_std=%s' % (it, rmses[-1], test_Y.std())
    #plt.plot(range(len(objs)), np.log10(objs), 'r-', range(len(objs)), rmses, 'g-')
    plt.plot(range(len(objs)), np.log10(objs), 'r-', range(len(objs)), rmses, 'g-')
    plt.show()
    return W

def load_data():
    diabetes = datasets.load_diabetes()
    data = diabetes.data

    # Split the data into training/testing sets
    X = data[:-20]
    test_X = data[-20:]

    # Split the targets into training/testing sets
    Y = diabetes.target[:-20]
    test_Y = diabetes.target[-20:]
    return X, Y, test_X, test_Y

def load_data_fromfile():
    filename = 'data.csv'
    data = np.loadtxt(filename, delimiter=',')
    #X = data[:-20,0]
    #X = X.reshape(X.size,1)
    #test_X = data[-20:,0]
    #test_X = test_X.reshape(test_X.size,1)
    #Y = data[:-20,1]
    #test_Y = data[-20:,1]
    return data[:,0].reshape(data.shape[0],1),data[:,1], data[:,0].reshape(data.shape[0],1),data[:,1]
    #return X, Y, test_X, test_Y

def lr():
    #X, Y, test_X, test_Y = load_data_fromfile()
    X, Y, test_X, test_Y = load_data()
    #import pdb;pdb.set_trace()
    print X.shape
    X = np.concatenate((np.ones((X.shape[0],1)), X), axis=-1)
    test_X = np.concatenate((np.ones((test_X.shape[0],1)), test_X), axis=-1)
    W = fit_lr(X, Y, test_X, test_Y)
    regr = linear_model.Ridge(alpha=0.0001, max_iter=1000,tol=1e-5)
    regr.fit(X, Y)
    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f" % np.mean((regr.predict(test_X) - test_Y) ** 2))
    print 'W=\n',W


if __name__ == '__main__':
    lr()
