from __future__ import division
import numpy as np
from scipy import io
from scipy.misc import logsumexp
import sys,os,re,math,scipy,glob,argparse, subprocess,json, time
import cPickle as pickle
from collections import defaultdict,Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from scipy.stats import beta
from sklearn import svm
import warnings 
from test_utils import *

"""
Goal: 
Print out predictions for 
(1) Baseline1 : predicting the training mean 
(2) Baseline2 : predicting 100% everywhere 
"""

def get_training_prevelance(trial, prop=False):
    #TODO: go through the training docs and get the true prevelance 
    if prop: 
        LOAD_TRAIN_PATH = '/home/kkeith/docprop/yelp_data/train2000_prop1/trial{0}/'.format(trial)
    else:
        LOAD_TRAIN_PATH = '/home/kkeith/docprop/yelp_data/train2000/trial{0}/'.format(trial) 
    trainY = np.load(LOAD_TRAIN_PATH+'trainY.npy')
    return np.mean(trainY)

def get_baseline1_mae(test_all_tys, prop=False):
    """
    for each trial get the training mean and use that as the map estimate
    """
    mae_across_trials = []
    bias_across_trials = []
    for trial in xrange(1, 11):
        map_est = get_training_prevelance(trial, prop=prop)
        mae = np.mean(np.abs(map_est - test_all_tys))
        mae_across_trials.append(mae)
        bias = np.mean(map_est- test_all_tys)
        bias_across_trials.append(bias)
    print 'bias (across trials) = ', np.mean(np.array(bias_across_trials))
    return np.mean(np.array(mae_across_trials))

def get_baseline2_mae(test_all_tys):
    map_est = 1.0
    mae = np.mean(np.abs(map_est - test_all_tys))
    print 'bias = ', np.mean(map_est- test_all_tys)
    return mae 

def get_py(inputcmd):
    labels = []
    for line in os.popen(inputcmd):
        d = json.loads(line)
        labels.append(d['class'])
    return np.array(labels)

def get_test_all_tys():
    TEST_PATH = "/home/kkeith/docprop/yelp_data/test/*"
    test_all_tys = []
    for f in sorted(glob.glob(TEST_PATH)):
        sys.stderr.write(".")
        ty = get_py("cat "+f)
        ymean_true = np.mean(ty)
        test_all_tys.append(ymean_true)
    test_all_tys = np.array(test_all_tys)
    assert len(test_all_tys) == 500 
    np.save('testY_all.npy', test_all_tys)
    return test_all_tys 

if __name__ == "__main__":
    #test_all_tys = get_test_all_tys()
    test_all_tys = np.load('testY_all.npy')
    print 'TEST MEAN=', np.mean(test_all_tys)

    mae = get_baseline1_mae(test_all_tys, prop=False)
    print 'MAE BASELINE 1 (natural prop):', mae 
    print
    mae =  get_baseline1_mae(test_all_tys, prop=True)
    print 'MAE BASELINE 1 (0.1 prop):', mae 
    print 
    mae = get_baseline2_mae(test_all_tys)
    print 'MAE BASELINE 2  (predict 1.0): ', mae 
    print 
