import scipy.io as spio
import numpy as np
import os
import copy

dir_ = 'data/epinions/'
# dir_ = ''
def whether_in(x,input_list):
	if x[0] in input_list:
		return True
	elif x[1] in input_list:
		return True
	else:
		return False

def generate_exp_split():
	rating = spio.loadmat(dir_+'rating.mat')

	# print rating['rating']
	# print rating['rating'][:,[0,1,3]]

	rating_filename = 'ratings'
	rating_ = rating['rating'][:,[0,1,3]]

	#remove duplicate entries
	rating_dict = {}
	for r in rating_:
		this_ub = str([r[0],r[1]])
		rating_dict[this_ub] = r[2]

	rating_rm = []
	for d,x in rating_dict.items():
		ub = eval(d)
		this_ = [ub[0],ub[1],x]
		rating_rm.append(this_)

	print(len(rating_rm))
	ratings = np.array(copy.deepcopy(rating_rm))
	# print ratings.shape[0]
	for n in xrange(5):
		exp_dir = dir_ + 'exp_split/%s/' % (n+1)
		if not os.path.isdir(exp_dir):
			os.makedirs(exp_dir)
			print 'create dir %s' % exp_dir
		train_filename = dir_ + 'exp_split/%s/%s.txt' % (n+1, rating_filename)
		test_filename = dir_ + 'exp_split/%s/test.txt' % (n+1)
		val_filename = dir_ + 'exp_split/%s/val.txt' % (n+1)

		rand_inds = np.random.permutation(ratings.shape[0])
		train_num = int(ratings.shape[0] * 0.8)
		test_num = int(ratings.shape[0]*0.1)
		train_data = ratings[rand_inds[:train_num]]
		test_data = ratings[rand_inds[train_num:(train_num+test_num)]]
		val_data = ratings[rand_inds[(train_num+test_num):]]
		np.savetxt(train_filename, train_data[:,:3], '%d\t%d\t%.1f')
		np.savetxt(test_filename, test_data[:,:3], '%d\t%d\t%.1f')
		np.savetxt(val_filename, val_data[:,:3], '%d\t%d\t%.1f')

		# generate user_social from train_data
		trustnetwork = spio.loadmat(dir_+'trustnetwork.mat')['trustnetwork']
		train_users = set(train_data[:,0])
		user_social_data = [x.tolist() for x in trustnetwork if whether_in(x,train_users)]
		user_social_filename = dir_ + 'exp_split/%s/user_social.txt' % (n+1)
		np.savetxt(user_social_filename, user_social_data, '%d\t%d')

		# generate uid_bid.txt from train_data
		uid_bid = train_data[:,:2]
		uid_bid_filename = dir_ + 'exp_split/%s/uid_bid.txt' % (n+1)
		np.savetxt(uid_bid_filename, uid_bid,'%d\t%d')

def generate_uids_and_bids_txt():
	rating = spio.loadmat(dir_+'rating.mat')['rating']
	trustnetwork = spio.loadmat(dir_+'trustnetwork.mat')['trustnetwork']

	parts = rating[:,[0,1]]
	uids_ = set([r[0] for r in parts]+[r[0] for r in trustnetwork]+[r[1] for r in trustnetwork])
	bids_ = set([r[1] for r in parts])
	uids = [[u] for u in uids_]
	bids = [[b] for b in bids_]

	for n in xrange(5):
		exp_dir = dir_ + 'exp_split/%s/' % (n+1)
		uids_filename = dir_ + 'exp_split/%s/uids.txt' % (n+1)
		bids_filename = dir_ + 'exp_split/%s/bids.txt' % (n+1)
		np.savetxt(uids_filename, uids, '%d')
		np.savetxt(bids_filename, bids, '%d')

if __name__ == '__main__':
    generate_exp_split()
    generate_uids_and_bids_txt()
