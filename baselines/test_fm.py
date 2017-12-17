import time
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from scipy.sparse import rand
from scipy.sparse import csr_matrix as cm

dt = 'yelp-50k'
t_dir = '../data/yelp-50k/exp_split/2/'
F = 10
N = 20
train_filename = t_dir + 'ratings_train_2.txt'
test_filename = t_dir + 'ratings_test_2.txt'

def load_rand_data():
    '''
        return the features, labels, and the group inds
    '''
    S, N = 10000, 400
    X = rand(10000, 400)
    Y = np.random.uniform(size=[S])

    test_X = rand(2000, N)
    test_Y = np.random.uniform(size=[2000])
    print 'train_data: (%.4f,%.4f), test_data: (%.4f,%.4f)' % (np.mean(Y), np.std(Y), np.mean(test_Y), np.std(test_Y))
    return X, Y, test_X, test_Y

def load_representation(t_dir, fnum, F):
    '''
        load user and item latent features generate by MF for every meta-graph
    '''
    if dt in ['yelp-200k', 'amazon-200k', 'amazon-50k', 'amazon-100k', 'amazon-10k', 'amazon-5k', 'cikm-yelp', 'yelp-50k', 'yelp-10k', 'yelp-5k', 'yelp-100k', 'douban']:
        ufilename = t_dir + 'uids.txt'
        bfilename = t_dir + 'bids.txt'
    uids = [int(l.strip()) for l in open(ufilename, 'r').readlines()]
    uid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in uids}

    bids = [int(l.strip()) for l in open(bfilename, 'r').readlines()]
    bid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in bids}

    #if dt == 'yelp-200k':
    #    ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBStarsB_top1000_user.dat', 'UPBStateB_top1000_user.dat', 'UPBCityB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'UUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
    #    vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBStarsB_top1000_item.dat', 'UPBStateB_top1000_item.dat', 'UPBCityB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'UUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']
    #if dt in ['yelp-10k', 'yelp-50k', 'yelp-100k', 'yelp-5k']:
    #    ufiles = ['URPSRUB_top500_user.dat', 'URNSRUB_top500_user.dat', 'UPBCatB_top500_user.dat', 'UPBStarsB_top500_user.dat', 'UPBStateB_top500_user.dat', 'UPBCityB_top500_user.dat', 'UPBUB_top500_user.dat', 'UNBUB_top500_user.dat', 'UUB_top500_user.dat', 'URPARUB_top500_user.dat', 'URNARUB_top500_user.dat']
    #    vfiles = ['URPSRUB_top500_item.dat', 'URNSRUB_top500_item.dat', 'UPBCatB_top500_item.dat', 'UPBStarsB_top500_item.dat', 'UPBStateB_top500_item.dat', 'UPBCityB_top500_item.dat', 'UPBUB_top500_item.dat', 'UNBUB_top500_item.dat', 'UUB_top500_item.dat', 'URPARUB_top500_item.dat', 'URNARUB_top500_item.dat']
    #elif dt == 'amazon-200k':
    #    ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBBrandB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
    #    vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBBrandB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']
    #elif dt in ['amazon-50k','amazon-100k','amazon-10k','amazon-5k']:
    #    ufiles = ['URPSRUB_top500_user.dat', 'URNSRUB_top500_user.dat', 'UPBCatB_top500_user.dat', 'UPBBrandB_top500_user.dat', 'UPBUB_top500_user.dat', 'UNBUB_top500_user.dat', 'URPARUB_top500_user.dat', 'URNARUB_top500_user.dat']
    #    vfiles = ['URPSRUB_top500_item.dat', 'URNSRUB_top500_item.dat', 'UPBCatB_top500_item.dat', 'UPBBrandB_top500_item.dat', 'UPBUB_top500_item.dat', 'UNBUB_top500_item.dat', 'URPARUB_top500_item.dat', 'URNARUB_top500_item.dat']
    #elif dt == 'cikm-yelp':
    #    ufiles = ['UPBCatBUB_top500_user.dat', 'UPBCityBUB_top500_user.dat','UNBCatBUB_top500_user.dat', 'UNBCityBUB_top500_user.dat', 'UPBUB_top500_user.dat', 'UNBUB_top500_user.dat', 'UUB_top500_user.dat', 'UCompUB_top500_user.dat']
    #    vfiles = ['UPBCatBUB_top500_item.dat', 'UPBCityBUB_top500_item.dat','UNBCatBUB_top500_item.dat', 'UNBCityBUB_top500_item.dat', 'UPBUB_top500_item.dat', 'UNBUB_top500_item.dat', 'UUB_top500_item.dat', 'UCompUB_top500_item.dat']
    #elif dt == 'douban':
    #    ufiles = ['UBDBUB_top500_user.dat', 'UBABUB_top500_user.dat', 'UBTBUB_top500_user.dat', 'UGUB_top500_user.dat', 'UBUB_top500_user.dat']
    #    vfiles = ['UBDBUB_top500_item.dat', 'UBABUB_top500_item.dat', 'UBTBUB_top500_item.dat', 'UGUB_top500_item.dat', 'UBUB_top500_item.dat']

    ufiles = ['ratings_only_user.dat']
    vfiles = ['ratings_only_item.dat']

    feature_dir = t_dir + 'mf_features/path_count/'
    #exp vary F
    if F != 10:
        feature_dir = t_dir + 'mf_features/path_count/ranks/'
    for find, filename in enumerate(ufiles):
        if F != 10:
            filename = filename.replace('user', 'F%s_user' % F)
            ufiles[find] = filename
        ufs = np.loadtxt(feature_dir + filename, dtype=np.float64)
        cur = find * F
        for uf in ufs:
            uid = int(uf[0])
            f = uf[1:]
            uid2reps[uid][cur:cur+F] = f

    for find, filename in enumerate(vfiles):
        if F != 10:
            filename = filename.replace('item', 'F%s_item' % F)
            vfiles[find] = filename
        bfs = np.loadtxt(feature_dir + filename, dtype=np.float64)
        cur = find * F
        for bf in bfs:
            bid = int(bf[0])
            f = bf[1:]
            bid2reps[bid][cur:cur+F] = f
    #logger.info('run for all, F=%s, len(ufiles)=%s, len(vfiles)=%s, ufiles=%s, vfiles=%s', len(ufiles), F, len(vfiles), '|'.join(ufiles), '|'.join(vfiles))

    return uid2reps, bid2reps

def load_data():
    start_time = time.time()

    train_data = np.loadtxt(train_filename)
    test_data = np.loadtxt(test_filename)
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]

    uid2reps, bid2reps = load_representation(t_dir, N/2, F)

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
        br = bid2reps.get(int(b), np.zeros(N/2, dtype=np.float64))
        test_X[ind] = np.concatenate((ur,br))
        ind += 1

    test_X_sparsity = np.count_nonzero(test_X) * 1.0 / test_X.size

    #logger.info('finish loading data, cost %.2f seconds, ratings_file=%s, train=%s, test=%s, stat(shape, sparsity): train: (%s, %.4f), test: (%s, %.4f)', time.time() - start_time, rating_filename, train_filename, test_filename, X.shape, X_sparsity, test_X.shape, test_X_sparsity)
    return X, Y, test_X, test_Y

def run():
    #X, Y, test_X, test_Y = load_rand_data()
    X, Y, test_X, test_Y = load_data()
    X = cm(X)
    test_X = cm(test_X)

    fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
    fm.fit(X, Y)
    pY = fm.predict(test_X)

    num = test_Y.shape[0]
    rmse = np.sqrt(np.square(pY - test_Y).sum() / num)
    print rmse

if __name__ == '__main__':
    run()
