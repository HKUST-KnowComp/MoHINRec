#coding=utf8
'''
    evalute exp result by giving the test dataset and filenames of learned parameters.
'''
import time
import numpy as np
import argparse

import yaml

from exp_util import cal_rmse, cal_mae
from data_util import DataLoader

split = 1
#dir_ = 'data/amazon-200k/exp_split/%s/' % split
test_filename = 'test_%s.txt' % split
#reg = 0.5 convex cikm-yelp
#W_filename = 'fm_res/cikm-yelp_split1_W_0.5_exp1514995488.txt'
#P_filename = 'fm_res/cikm-yelp_split1_P_0.5_exp1514995488.txt'

W_filename = 'fm_res/yelp-200k_split1_W_0.5_exp1514957804.txt'
P_filename = 'fm_res/yelp-200k_split1_P_0.5_exp1514957804.txt'

#W_filename = 'fm_res/amazon-200k_split1_W_0.05_exp1514989178.txt'
#P_filename = 'fm_res/amazon-200k_split1_P_0.05_exp1514989178.txt'

#reg=0.5, non-convex
#W_filename = 'fm_res/yelp-200k_split1_W_0.5_exp1514996292.txt'
#P_filename = 'fm_res/yelp-200k_split1_P_0.5_exp1514996292.txt'

#reg=0.5, non-convex
#W_filename = 'fm_res/amazon-200k_split1_W_0.5_exp1514996455.txt'
#P_filename = 'fm_res/amazon-200k_split1_P_0.5_exp1514996455.txt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', help='specify the dataset for exp, e.g., yelp-50k')
    parser.add_argument('-K', help='number of latent features when factorizing P, or Q in FM', type=int)
    parser.add_argument('-reg', help='regularization for all parameters, if given, set all reg otherwise doing nothing', type=float)
    parser.add_argument('-solver', help='specify the opt solver, e.g., nmAPG, svrg')
    parser.add_argument('-reg_P', help='regularization for P', type=float)
    parser.add_argument('-reg_Q', help='regularization for Q', type=float)
    parser.add_argument('-reg_W', help='regularization for W', type=float)
    parser.add_argument('-max_iters', help='max iterations of the training process', type=int)
    parser.add_argument('-eps', help='stopping criterion', type=float)
    parser.add_argument('-eta', help='learning rate in the beginning', type=float)
    parser.add_argument('-bias_eta', help='learning rate for bias', type=float)
    parser.add_argument('-initial', help='initialization of random starting', type=float)
    parser.add_argument('-nnl', help='lambda in nuclear norm, currently it denotes the type of lambda', type=int)
    parser.add_argument('config',  help='specify the config file')
    parser.add_argument('wf',  help='specify the filename of W')
    parser.add_argument('vf',  help='specify the filename of V')
    return parser.parse_args()

def update_configs_by_args(config, args):
    args_dict = vars(args)
    #if reg is specified, set all regularization values to reg
    if args.reg is not None:
        config['reg_W'] = config['reg_P'] = config['reg_Q'] = args.reg
        del args_dict['reg_W']
        del args_dict['reg_P']
        del args_dict['reg_Q']

    for k, v in args_dict.items():
        if v is not None:
            config[k] = v

def update_configs(config, args):
    '''
        1, generate some configs dynamically, according to given parameters
            L, N, exp_id, logger
        2, fix one bug: make 1e-6 to float
        3, create exp data dir, replacing 'dt' with the specified dt
        3, update by arguments parser
    '''
    exp_id = int(time.time())
    config['exp_id'] = exp_id


    L = len(config.get('meta_graphs'))
    config['L'] = L

    F = config['F']
    config['N'] = 2 * L * F

    update_configs_by_args(config, args)
    config['eps'] = float(config['eps'])
    config['initial'] = float(config['initial'])
    config['eta'] = float(config['eta'])
    config['bias_eta'] = float(config['bias_eta'])
    dt = config['dt']
    config['data_dir'] = config.get('data_dir').replace('dt', dt)

def init_exp_configs(config_filename):
    '''
        load the configs
    '''
    config = yaml.load(open(config_filename, 'r'))
    config['config_filename'] = config_filename
    return config

def generate_testXY(test_data, uid2reps, bid2reps, N):

    test_num = test_data.shape[0]

    test_X = np.zeros((test_num, N))
    test_Y = test_data[:,2]

    ind = 0
    for u, b, _ in test_data:
        ur = uid2reps.get(int(u), np.zeros(N/2))
        br = bid2reps.get(int(b), np.zeros(N/2))
        test_X[ind] = np.concatenate((ur,br))
        ind += 1

    return test_X, test_Y

def cal_err_by_data(bias, W, P, X, Y):
    '''
        return an error vector of (pred - obs), given the data, e.g., train_X, mini_batch_X, etc
    '''
    WX, XP, XSPS = get_XC_prods(X, W, P)
    Y_t = bias + WX + 0.5 * (np.square(XP) - XSPS).sum(axis=1)
    return Y_t - Y

def get_XC_prods(X, W, P):
    WX = np.dot(W, X.T)
    XP = np.dot(X, P)
    XSPS = np.dot(np.square(X), np.square(P))
    return WX, XP, XSPS

def get_bias(train_file):
    '''
        get bias from train_data
    '''
    data = np.loadtxt(train_file)
    return np.mean(data[:,2])

def run(args):
    config = init_exp_configs(args.config)
    update_configs(config, args)
    data_loader = DataLoader(config)

    uid2reps, bid2reps = data_loader._load_representation()

    data_dir = config['data_dir']
    train_filename = config['train_filename']
    test_data = np.loadtxt(data_dir + test_filename)
    N = config['N']
    test_X, test_Y = generate_testXY(test_data, uid2reps, bid2reps, N)
    W = np.loadtxt(args.wf)
    P = np.loadtxt(args.vf)
    bias = get_bias(data_dir + train_filename)
    test_err = cal_err_by_data(bias, W, P, test_X, test_Y)
    print cal_rmse(test_err), cal_mae(test_err)

if __name__ == '__main__':
    args = get_args()
    run(args)
