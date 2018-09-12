from __future__ import division
import numpy as np
from scipy import io
from scipy import sparse
import scipy.misc
import sys,os,re,math,scipy
import json as ujson
from scipy.misc import logsumexp
import cPickle as pickle
from collections import defaultdict,Counter
from sklearn.feature_extraction import DictVectorizer

############################################

def load_to_matrix(inputcmd, word2num):
    import scipy.sparse as ss
    labels = []
    V = len(word2num)
    rows = []
    for line in os.popen(inputcmd):
        counts = np.zeros(V, dtype=int)
        d = ujson.loads(line)
        wc = d['counts']
        labels.append(d['class'])
        for w,c in wc.iteritems():
            if w not in word2num: continue
            counts[word2num[w]] = c
        rows.append(ss.csr_matrix(counts))
    return ss.vstack(rows), np.array(labels)

def safe_toarray(x):
    """Intended to take a matrix or sparse matrix that's only one row and
    return a 1d array.  for a matrix this is just shifting the view (probably).
    for a sparse matrix i think it actually allocates new memory"""

    if isinstance(x, scipy.sparse.spmatrix):
        assert x.shape[0]==1 or x.shape[1]==1
        return np.asarray(x.todense()).flatten()
    if isinstance(x, np.matrix):
        assert x.shape[0]==1 or x.shape[1]==1
        return np.asarray(x).flatten()
    if isinstance(x,np.ndarray):
        return x
    assert False
    return x

def dotlog(x,p):
    """return x'log(p) but don't evaluate log(p) where x=0
    x and p are 1d arrays of same length"""
    ll = 0.0
    n = len(x)
    for i in xrange(n):
        if x[i]==0: continue
        if p[i]==0:
            ll += -np.inf
            continue
        ll += x[i]*np.log(p[i])
    return ll

def sparsedotlog(sparsex, p):
    """return x'log(p) but don't evaluate log(p) where x=0
    sparsex: assumed to be a (1 x N) csr_matrix (may work with (N x 1) csc??)
    p: a dense array length N
    """
    ll = 0.0
    # xrange vs zip: nearly identical runtime
    for i in xrange(sparsex.nnz):
        reali = sparsex.indices[i]
        xvalue = sparsex.data[i]
        if p[reali]==0:
            ll += -np.inf
            continue
        ll += xvalue * np.log(p[reali])
    return ll

##########################################################

# *curve() functions: return a score/LL for every grid point on 0 to 1
# xentcurve(), sqcurve() take testset aggregated counts as input
# mnb, dcm take a matrix representing indiv documents
# 'mll' means log(marginal likelihood)

trange = np.arange(0.001, 1.0, 0.001)
#trange = np.arange(0.1, 1.0, 0.1)

def xentcurve(testcounts, w_given_class):
    """better to call it: marginal LM crossent minimization
    old code called this nllcurve()"""
    testcounts = safe_toarray(testcounts)
    # maybe should call it a "marginal crossentropy curve"
    ll_list = []
    for t in trange:
        ll = testcounts.dot(np.log(w_given_class.dot([1-t,t])))
        ll_list.append(ll)
    # to actually get xent divide by testcounts.sum()
    return -np.array(ll_list)

def sqcurve(testcounts, w_given_class):
    testcounts = safe_toarray(testcounts)
    Ntok = testcounts.sum()
    err_list = []
    for t in trange:
        predcounts = Ntok*w_given_class.dot([1-t,t])
        err = np.sum( (testcounts - predcounts)**2 )
        err_list.append(err)
    return np.array(err_list)

####################

def mnb_ll_onedoc(countvec, w_probs):
    return sparsedotlog(countvec, w_probs)
    #return dotlog(countvec, w_probs)

def mnb_mll_curve_onedoc(countvec, w_given_class):
    # evaluate likelihood only once
    lls = [mnb_ll_onedoc(countvec, w_given_class[:,k]) for k in [0,1]]
    mlls = [scipy.misc.logsumexp(lls + np.log([1-t,t])) for t in trange]
    return np.array(mlls)

def mnb_mll_curve(dtm, w_given_class):
    curve = np.zeros(len(trange))
    for doc in dtm:
        curve += mnb_mll_curve_onedoc(doc, w_given_class)
    return curve

def mnb_onedoc_pred(countvec, w_given_class, posprior=0.5):
    lls = [mnb_ll_onedoc(countvec, w_given_class[:,k]) for k in [0,1]]
    relative_lprob = np.array(lls) + np.log([1-posprior,posprior])
    logposts = relative_lprob-logsumexp(relative_lprob)
    print logposts
    return np.exp(logposts[1])

def mnb_log_onedoc_pred(countvec, w_given_class, posprior=0.5):
    """ 
    returns the log posterior 
    """
    lls = [mnb_ll_onedoc(countvec, w_given_class[:,k]) for k in [0,1]]
    relative_lprob = np.array(lls) + np.log([1-posprior,posprior])
    logposts = relative_lprob-logsumexp(relative_lprob)
    return logposts

##################################################################

##KAK: aggregating individual MLL estimates

def softmax(x):
    """
    numerically stable softmax function 
    """
    x = x - np.max(x)
    return np.exp(x)/ np.sum(np.exp(x))

def indvl_get_py(tx, w_given_class):
    """
    for each group get the probability of each class 
    """
    py = []
    for doc in tx: 
        lls = [mnb_ll_onedoc(doc, w_given_class[:,k]) for k in [0,1]]
        #take the softmax over those log likelihoods
        probs = softmax(lls)
        py.append(probs[1]) #append the positive class 
    assert len(py) == tx.shape[0]
    return np.array(py)

def indvl_get_log_py(tx, w_given_class, posprior=0.5):
    """
    for each group get the log probabilites 
    """
    logp = []
    for doc in tx: 
        logp.append(mnb_log_onedoc_pred(doc, w_given_class, posprior=posprior))
    assert len(logp) == tx.shape[0]
    return np.array(logp)

def test_indvl_get_log_py():
    import pytest 

    #just make sure the thing runs w/ no syntax errors 
    tx = sparse.csr_matrix(np.array(
        [[2, 3, 4, 0],
        [0, 1, 2, 3],
        [2, 0, 0, 1]]))
    w_given_class = np.ones((4, 2))
    posprior = 0.7
    print indvl_get_log_py(tx, w_given_class, posprior)
    #assert 0 == 1 

###############################

##KAK: Hybrid, implicit generative model

#TODO: could vectorize this and it would be A LOT faster... 

def hybrid_mll_curve(p_neg_x, p_pos_x, pi):
    """
    Parameters:
        p_neg_x : numpy array, size (ndocs,)
            P(y=0 | w) given by logistic regression (or other discriminative model)

        p_pos_x : numpy array, size (ndocs,)
            P(y=1 | w) given by logistic regression (or other discriminative model)

        pi: prevelance of positive class
            estimated from the training data 
            
    Returns:
        mll_curve: numpy array, size (len(trange),) 

    """
    assert p_neg_x.shape == p_pos_x.shape
    curve = np.zeros(len(trange))
    for p_neg_x_i, p_pos_x_i in zip(p_neg_x, p_pos_x):
        curve = curve + hybrid_mll_curve_onedoc(p_neg_x_i, p_pos_x_i, pi)
    return curve

def hybrid_mll_curve_onedoc_onetheta(p_neg_x_i, p_pos_x_i, pi, theta):
    term = 1.0 - theta + (p_pos_x_i/p_neg_x_i)*((1.0-pi)/pi)*theta
    return np.log(term)

def hybrid_mll_curve_onedoc(p_neg_x_i, p_pos_x_i, pi):
    curve = np.array([hybrid_mll_curve_onedoc_onetheta(p_neg_x_i, p_pos_x_i, pi, theta) for theta in trange])
    return curve 

def test_hybrid_one_theta():
    import pytest
    p_neg_x_i = 0.9
    p_pos_x_i = 0.1
    theta = 0.1
    pi = 0.5
    assert np.round(hybrid_mll_curve_onedoc_onetheta(p_neg_x_i, p_pos_x_i, pi, theta), 3) == -0.093 
    #assert 0 == 1

def test_hybrid():
    import pytest

    p1 = np.array([0.9, 0.9, 0.9])
    p2 = np.array([0.1, 0.1, 0.1])
    curve = hybrid_mll_curve(p1, p2, 0.5)
    assert trange[np.argmax(curve)] == 0.001

    p1 = np.array([0.1, 0.1, 0.1])
    p2 = np.array([0.9, 0.9, 0.9])
    curve = hybrid_mll_curve(p1, p2, 0.5)
    assert trange[np.argmax(curve)] == 0.999 

    p1 = np.array([0.9, 0.9, 0.9])
    p2 = np.array([0.1, 0.1, 0.1])
    curve = hybrid_mll_curve(p1, p2, 0.9)
    assert trange[np.argmax(curve)] == 0.001

    #truly no information between the two, should get 0's everywhere, then our generative map est will take the median 
    p1 = np.array([0.5, 0.5, 0.5])
    p2 = np.array([0.5, 0.5, 0.5])
    curve = hybrid_mll_curve(p1, p2, 0.5)
    assert np.max(curve) == 0.0

    p1 = np.array([0.6, 0.3, 0.6])
    p2 = np.array([0.4, 0.6, 0.4])
    curve = hybrid_mll_curve(p1, p2, 0.5)
    assert np.round(trange[np.argmax(curve)], 3) == 0.333

#numerically stable versions 

def hybrid_mll_curve_stable(logodds, pi):
    """
    Parameters:
        logodds: nparray size (ndocs, )
            log(p(y=1 | w) / p(y=0 | w))
            = wx + b
            (which comes out of sklearn logistic regression decision_function)

        pi: prevelance of positive class
            estimated from the training data 
            
    Returns:
        mll_curve: numpy array, size (len(trange),) 

    """
    curve = np.zeros(len(trange))
    log_pi_ratio = np.log((1.0 - pi)/ pi)
    for logodds_onedoc in logodds:
        curve += hybrid_mll_curve_onedoc_stable(logodds_onedoc, log_pi_ratio)
    return curve

def hybrid_mll_curve_onedoc_onetheta_stable(logodds_onedoc, log_pi_ratio, theta):
    term = 1.0 - theta + np.exp(np.log(theta) + logodds_onedoc + log_pi_ratio)
    return np.log(term)

def hybrid_mll_curve_onedoc_stable(logodds_onedoc, log_pi_ratio):
    curve = np.array([hybrid_mll_curve_onedoc_onetheta_stable(logodds_onedoc, log_pi_ratio, theta) for theta in trange])
    return curve 

def test_hybrid_stable():
    import pytest
    p1 = 0.9
    p2 = 0.1
    logodds = np.log(p2/p1)
    pi = 0.3
    log_pi_ratio = np.log((1.0 - pi)/pi)
    assert hybrid_mll_curve_onedoc_onetheta(p1, p2, pi, 0.1) == hybrid_mll_curve_onedoc_onetheta_stable(logodds, log_pi_ratio, 0.1)

    p1 = np.array([0.6, 0.3, 0.6])
    p2 = np.array([0.4, 0.6, 0.4])
    logodds = np.log(p2/p1)
    pi = 0.5 
    assert np.allclose(hybrid_mll_curve(p1, p2, pi), hybrid_mll_curve_stable(logodds, pi))

#TODO: could vectorize these and make them MUCH faster! 

