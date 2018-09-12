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

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'utils'))
import infer, ecdf

def roundprec(x,reso):
    y = reso*np.round(x/reso)
    y = np.clip(y,0,1)
    return y

def get_lr_samples(py, S=10000):
    '''
    Poisson-binomial MC sampler for logreg soft (PCC)
    draw a bernoulli sample and find the mean 
    repeat S times to get a sample PMF  
    py :: array of probabilites of postive class; size of num docs
    '''
    counts = Counter()
    for s in xrange(S):
        y_draw = np.random.binomial(1, py)
        prop_est = np.mean(y_draw)
        counts[prop_est] += 1
    return counts

def get_test_prior():
    '''
    (1) "cheating" beta prior from the empirical test distribution
        get this by first running beta_test_prior.py 
    '''
    log_prior = np.load('test_prior_log_beta.npy')
    assert len(log_prior) == len(infer.trange)
    return log_prior 

def get_beta_prior():
    '''
     (2) Beta(1+eps, 1+eps)
        we need to add this prior to the mll probs in the case where 
        we have a flat mll curve('i.e. high regularization')
    '''
    #prior (2)
    eps = 0.0001
    a, b = 1.0+eps, 1.0+eps
    gridpoints = np.linspace(0.001, 0.999, 999)
    log_prior = beta.logpdf(gridpoints, a, b) 
    assert len(log_prior) == len(infer.trange)
    return log_prior 

def load_model(model_type, num_train, reg_strength, has_betaAllTest):
    if model_type == 'logreg':
        return joblib.load('../other_models/models{0}/lrm_{1}.pkl'.format(num_train, reg_strength))
    elif model_type == 'mnb':
        return np.load('../other_models/models{0}/mnb_{1}.npy'.format(num_train, reg_strength))
    elif model_type == 'loglin':
        if has_betaAllTest:
            return np.load('../loglin/models{0}/loglin_{1}_alltest.npy'.format(num_train, float(reg_strength)))
        else: 
            return np.load('../loglin/models{0}/loglin_{1}.npy'.format(num_train, float(reg_strength)))
    raise Exception('invalid model input')

def load_model_trainprop(model_type, num_train, reg_strength, trainprop):
    if model_type == 'logreg':
        return joblib.load('../other_models/trainprop/lrm_{0}_prop{1}.pkl'.format(reg_strength, trainprop))
    elif model_type == 'mnb':
        return np.load('../other_models/trainprop/mnb_{0}_prop{1}.npy'.format(reg_strength, trainprop))
    elif model_type == 'loglin':
        return np.load('../loglin/trainprop/loglin_{0}_prop{1}.npy'.format(float(reg_strength), trainprop))
    raise Exception('invalid model input')

def load_vocab(num_train, trainprop=None):
    PATH='../../yelp_data/train{}/'.format(num_train)
    if trainprop != None:
        PATH += 'prop{0}/'.format(trainprop)
    with open(PATH+'word2num.json', 'r') as r1: 
        word2num = json.load(r1)
    return word2num

def generative_get_map_est(log_post_probs):
    '''
    returns the map estimate of the posterior for generative models 

    in the case of ties for the max posterior we take the middle index
    '''
    assert len(log_post_probs) == len(infer.trange)

    #check multi-modal posterior!  
    mx = np.max(log_post_probs)
    if len(log_post_probs[log_post_probs == mx]) >= 2:
        #ipdb.set_trace() 
        #raise Exception("You have a multimodal posterior distribution!")
        where = (np.where(log_post_probs == mx))[0]
        warnings.warn("You have a multimodal posterior distribution where "+str(where)+' Resolving tie by picking the middle value.')
        middle_index = int(np.floor(len(where)/2))
        map_est = infer.trange[where[middle_index]]

    else: 
        map_est = infer.trange[np.argmax(log_post_probs)]
    
    return map_est

def test_generative_get_map_est():
    import pytest
    #tie 
    log_post_probs = np.zeros(len(infer.trange))
    log_post_probs[0:6] = np.array([10, 11, 20, 20, 20, 14])
    map_est = generative_get_map_est(log_post_probs)
    assert  map_est== 0.001*4

    #no tie
    log_post_probs = np.zeros(len(infer.trange))
    log_post_probs[0:4] = np.array([10, 11, 20, 14])
    map_est = generative_get_map_est(log_post_probs)
    assert  map_est== 0.001*3