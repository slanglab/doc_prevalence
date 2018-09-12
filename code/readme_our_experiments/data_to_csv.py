from __future__ import division
import json 
import numpy as np 
import scipy
from scipy import sparse
from sklearn.externals import joblib

def load_and_save_train_data(LOAD_PATH, SAVE_PATH):
    trainX = scipy.sparse.load_npz(LOAD_PATH+'trainX.npz')
    trainY = np.load(LOAD_PATH+'trainY.npy')
    np.savetxt(SAVE_PATH+'trainX.csv', trainX.toarray().astype(int), delimiter=',', fmt='%i')
    np.savetxt(SAVE_PATH+'trainY.csv', trainY.astype(int), delimiter=',', fmt='%i')

def change_train_data_to_csv():
    print 'TRAIN'
    for trial in xrange(1, 11):
        print trial 
        #natural 
        LOAD_PATH = '/home/kkeith/docprop/yelp_data/train2000/trial'+str(trial)+'/'
        SAVE_PATH = 'train/nat/trial'+str(trial)+'/'
        load_and_save_train_data(LOAD_PATH, SAVE_PATH)
        #synthetic 
        LOAD_PATH = '/home/kkeith/docprop/yelp_data/train2000_prop1/trial'+str(trial)+'/'
        SAVE_PATH = 'train/prop1/trial'+str(trial)+'/'
        load_and_save_train_data(LOAD_PATH, SAVE_PATH)

def load_and_save_test_data(LOAD_PATH, SAVE_PATH):
    loaded_testdata = joblib.load(LOAD_PATH)
    i = 1 
    for tx, ty in zip(loaded_testdata['all_xs'], loaded_testdata['all_ygold']):
        np.savetxt(SAVE_PATH+'group{0}X.csv'.format(i), tx.toarray().astype(int), delimiter=',', fmt='%i')
        np.savetxt(SAVE_PATH+'group{0}Y.csv'.format(i), ty.astype(int), delimiter=',', fmt='%i')
        i += 1

def change_test_data_to_csv():
    print 'TEST'
    for trial in xrange(1, 11): 
        print trial 
        #natural
        LOAD_PATH = '../eval/saved_testdata/trial{0}.joblib'.format(trial)
        SAVE_PATH = 'test/nat/trial{0}/'.format(trial) 
        load_and_save_test_data(LOAD_PATH, SAVE_PATH)

        #synthetic 
        LOAD_PATH = '../eval/saved_testdata/trial{0}_prop1.joblib'.format(trial)
        SAVE_PATH = 'test/prop1/trial{0}/'.format(trial) 
        load_and_save_test_data(LOAD_PATH, SAVE_PATH)

if __name__ == "__main__":
    #change_train_data_to_csv()
    change_test_data_to_csv()

