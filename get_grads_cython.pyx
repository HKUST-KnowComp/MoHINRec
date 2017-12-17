import numpy as np
from cython import boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
def get_grad_v(double [:] err, double [:] W, double [:,:] V, double [:,:] X, double [:,:] part, double [:,:] X_square, int N, int K, int M):
    #grad_b = 2 * np.sum(err)
    #grad_w = 2 * np.dot(err, X)

    cdef double tmp;
    #cdef double [:,:] part = np.dot(X, V)
    cdef double [:,:] grad_v = np.zeros((N,K))
    #cdef double [:,:] X_square = np.square(X)
    cdef int i, j, k
    for i in range(N):
        for f in range(K):
            tmp = 0.0
            for k in range(M):
                tmp += err[k] * (X[k,i] * part[k,f] - V[i,f] * X_square[k,i])
            grad_v[i,f] = 2 * tmp
            #grad_v[i,f] = 2 * np.dot(err, np.multiply(X[:,i], part[:,f]) - V[i,f] * X_square[:,i])
    #return grad_b, grad_w, grad_v
    return np.asarray(grad_v)

