from __future__ import division
import json 
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from collections import defaultdict, Counter
from lbfgs import LBFGS, LBFGSError, fmin_lbfgs
from scipy import io, sparse

def cross_ent_stable(y, logodds):
    '''
    issues with numerical stability of cross entropy

    if ytrue ==1 and p==0. logp = -inf and we have a problem 
    if we clip p[1e-10] = 1e-10 this is also a problem b/c the size of the clipping will
        drastically change the cross-entropy calculations

    solution: 
    pass in the logodds ratio log(P(A)/ (1-P(A))
    this logodds ratio is also the same as what comes out of sklearn logistic regression decision_function(X)
        
    Parameters
    ------
    D = number of instances 

    y: (D, 1) numpy array, true values of y 

    logodds: (D, ) numpy array of the log odds or the output of the LogisticRegression.decision_function
        which is equal to WX + b

    returns TOTAL entropy (you will have to manually divide to get average later on...)
    '''

    #if True in np.isinf(logp): raise ValueError('the logprobs have an inf or -inf value!')

    crossent = 0 
    D = y.shape[0]
    #assert D == logp.shape[0]
    assert D == logodds.shape[0]
    for i in xrange(D):
        #logodds = logp[i][1] - logp[i][0] #log(p/1-p) where p is pos class
        if y[i] == 1: 
            crossent += -1.0*scipy.special.logsumexp([0, -1.0*logodds[i]])
        elif y[i] == 0 :
            crossent += -1.0*scipy.special.logsumexp([0, logodds[i]])
        else: 
            raise ValueError('Your y array does not have exclusively 1s and 0s!')

    return -1.0*crossent

def make_cv_folds(trainX, trainY, num_folds=10):
    NUM_TRAIN_DOCS = trainX.shape[0]
    assert NUM_TRAIN_DOCS == len(trainY)

    #shuffle the indexes 
    idxs = np.arange(NUM_TRAIN_DOCS)
    np.random.shuffle(idxs)

    num_in_group = int(NUM_TRAIN_DOCS/ num_folds)

    #make the cross validation train and dev groups 
    groups = {}
    for k in xrange(num_folds):
        #leave one out 
        start = k * num_in_group
        end = start + num_in_group

        devX = trainX[idxs[start:end]]
        devY = trainY[idxs[start:end]]

        assert len(devX) == len(devY) == num_in_group

        cross_idxs = np.hstack((idxs[0:start], idxs[end:(len(idxs))]))
        crossX = trainX[cross_idxs]
        crossY = trainY[cross_idxs]

        assert len(crossX) == len(crossY) == NUM_TRAIN_DOCS - num_in_group

        groups[k] = {'train': (crossX, crossY), 'dev': (devX, devY)} 
    return groups 

