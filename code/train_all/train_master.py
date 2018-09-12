from __future__ import division
import sys, os, warnings
import numpy as np
from scipy import io
from scipy.misc import logsumexp
import sys,os,re,math,scipy,glob,argparse, subprocess, json
import cPickle as pickle
from collections import defaultdict,Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from scipy import io, sparse
from lbfgs import LBFGS, LBFGSError, fmin_lbfgs
import crossval_utils

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'utils'))
import infer

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'loglin'))
import model as loglinmm

#import ipdb

np.random.seed(seed=12345)

def train_mnb(pseudo, w_class_count, V):
    wcc = w_class_count + pseudo*1.0/V
    tot_class_count=wcc.sum(0)
    w_given_class = (wcc / tot_class_count)
    return w_given_class

def train_loglin(trainX, trainY, reg_const):
    D, V = trainX.shape
    K = 2 #two classes 
    #print 'K={0}, D={1}, V={2}'.format(K, D, V)
    assert trainX.shape[0] == len(trainY)
    beta = loglinmm.get_beta(trainX)
    gamma0 = np.random.rand(K, V)
    
    #PYLBFGS stuff 
    bb=LBFGS()
    bb.orthantwise_c = 2.0**reg_const
    bb.linesearch = 'wolfe'
    bb.epsilon = 1e-01
    bb.delta = 1e-01

    #find optimum 
    gamma_opt = bb.minimize(loglinmm.negll, gamma0, None, [beta, trainX, trainY])
    w_given_class = loglinmm.prob_w_given_k(beta, gamma_opt)
    return w_given_class


def get_w_class_count(trainX, trainY):
    w_class_count = np.vstack(
        [
            np.asarray(trainX[trainY==0].sum(0)).flatten(),
            np.asarray(trainX[trainY==1].sum(0)).flatten(),
        ]).T
    w_class_count=np.array(w_class_count)
    return w_class_count

def get_cross_val_splits(trainX, trainY, num_folds=10):
    """
    10-fold cross validation 
    """
    groups = crossval_utils.make_cv_folds(trainX, trainY, num_folds=num_folds)

    for fold in xrange(num_folds):
        w_class_count = get_w_class_count(groups[fold]['train'][0], groups[fold]['train'][1])
        groups[fold]['w_class_count'] = w_class_count

    return groups

def get_tpr_fpr(y_true, y_pred):
    """
    Returns: 
        tpr (true positive rate)
            -same as recall
            true_positives / (true_positives + false_negatives)
        fpr (false positive rate)
            false_positives / (false_positives + true negatives)
    """
    assert len(y_true) == len(y_pred)
    # print len(y_true), len(y_pred)
    # print y_true, y_pred

    #case where they are all the same, we get an error out of the confusion matrix 
    if np.all(y_true == y_pred): 
        return 1.0, 0.0 

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #print tn, fp, fn, tp
    if (tp + fn) == 0: 
        true_pos_rate = 0.0
        warnings.warn('WARNING: tp + fn == 0')
    else: true_pos_rate = tp / (tp + fn)
    if (fp + tn) == 0:
        false_pos_rate = 0.0 
        warnings.warn('WARNING: fp + tn == 0')
        #ipdb.set_trace()
    else: false_pos_rate = fp / (fp + tn)
    return true_pos_rate, false_pos_rate

def test_get_tpr_fpr():
    import pytest 

    ytrue =np.array([1, 1, 1, 0, 0])
    ypred = np.array([1, 1, 0, 1, 0])
    tpr, fpr = get_tpr_fpr(ytrue, ypred)
    assert tpr == 2.0/3.0
    assert fpr == 1.0/2.0 

def get_best_reg(reg2xent, model_class):
    """
    Returns: 
        best_reg : regularization with the LOWEST mean cross entropy 

    """
    print 'CROSS-VAL, ', model_class
    print '=='*10
    best_xent = 10**10
    best_reg = None
    for reg, xent_list in sorted(reg2xent.iteritems()):
        mean_xent = np.mean(np.array(xent_list))
        print 'reg={0}, mean_xent={1}'.format(reg, mean_xent)
        if mean_xent < best_xent:
            best_xent = mean_xent
            best_reg = reg 

    print '--'*10
    print 'best_reg={0}, best_xent={1}'.format(best_reg, best_xent)
    print '=='*10
    return best_reg

def test_get_best_reg():
    import pytest 
    reg2xent = {0: [.3, .4, .5], 1: [.6, .7, .8]}
    best_reg = get_best_reg(reg2xent, 'logreg')
    assert best_reg == 0 

def get_best_logreg_reg_via_cross_val(cross_val_groups, num_folds=10):
    reg_values = np.arange(-13, 12)

    reg2xent = defaultdict(list)
    reg2tpr = defaultdict(list)
    reg2fpr = defaultdict(list)

    for reg in reg_values:
        lrm=LogisticRegression(penalty='l1', C=1.0/2.0**reg, verbose=0) 
        for fold in xrange(num_folds):
            crossX, crossY = cross_val_groups[fold]['train']
            devX, devY = cross_val_groups[fold]['dev']
            lrm.fit(crossX, crossY)
            logodds = lrm.decision_function(devX)
            #TODO: I don't think this is average cross entropy across all docs but this should be 
            #ok b/c all folds have the same number of docs??  
            xent = crossval_utils.cross_ent_stable(devY, logodds)
            reg2xent[reg].append(xent)

            #tpr, fpr for ACC method
            predY = lrm.predict(devX)
            tpr, fpr = get_tpr_fpr(devY, predY)
            reg2tpr[reg].append(tpr)
            reg2fpr[reg].append(fpr)

    best_reg = get_best_reg(reg2xent, 'logreg')

    best_reg_tpr = np.mean(np.array(reg2tpr[best_reg]))
    best_reg_fpr = np.mean(np.array(reg2fpr[best_reg])) 

    print 'TPR={0}, FPR={1}'.format(best_reg_tpr, best_reg_fpr)

    return best_reg, best_reg_tpr, best_reg_fpr 

def get_best_loglin_reg_via_cross_val(cross_val_groups, num_folds=10, posprior=0.5):
    reg_values = np.arange(-9, 12)
    reg2xent = defaultdict(list)

    for reg in reg_values:
        for fold in xrange(num_folds):
            crossX, crossY = cross_val_groups[fold]['train']
            devX, devY = cross_val_groups[fold]['dev']
            devX = sparse.csr_matrix(devX)
            loglin_wgc = train_loglin(crossX, crossY, reg)
            logp = infer.indvl_get_log_py(devX, loglin_wgc, posprior=posprior)
            logodds = logp[:, 1] - logp[:, 0]
            xent = crossval_utils.cross_ent_stable(devY, logodds)
            reg2xent[reg] = xent

    best_reg = get_best_reg(reg2xent, 'loglin')
    return best_reg

def get_best_mnb_reg_via_cross_val(cross_val_groups, num_folds=10, posprior=0.5):
    reg_values = np.arange(1, 18)
    reg2xent = defaultdict(list)

    for reg in reg_values:
        for fold in xrange(num_folds):
            crossX, crossY = cross_val_groups[fold]['train']
            devX, devY = cross_val_groups[fold]['dev']
            devX = sparse.csr_matrix(devX)
            w_class_count = cross_val_groups[fold]['w_class_count']
            V = crossX.shape[1]
            wgc = train_mnb(2.0**reg, w_class_count, V)
            logp = infer.indvl_get_log_py(devX, wgc, posprior=posprior)
            logodds = logp[:, 1] - logp[:, 0]
            xent = crossval_utils.cross_ent_stable(devY, logodds)
            reg2xent[reg] = xent

    best_reg = get_best_reg(reg2xent, 'mnb')
    return best_reg 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_train", help="number of training documents", type=int)
    parser.add_argument("model_class", help="model class (either mnb, loglin, or logreg)", type=str)
    parser.add_argument("trial", help="trial 1 thru 10 ", type=int)
    parser.add_argument("--prop", type=int, default=None)
    args = parser.parse_args()

    if args.prop != None: PATH='../../yelp_data/train{0}_prop{1}/trial{2}/'.format(args.num_train, args.prop, args.trial)
    else: PATH='../../yelp_data/train{0}/trial{1}/'.format(args.num_train, args.trial)

    print 'ntrain={0}, model={1}, trial={2}'.format(args.num_train, args.model_class, args.trial)
    print 'TRAIN PREV=', args.prop 

    trainX = scipy.sparse.load_npz(PATH+'trainX.npz').toarray()
    trainY = np.load(PATH+'trainY.npy')
    with open(PATH+'word2num.json', 'r') as r1: 
        word2num = json.load(r1)

    train_posprior = np.mean(trainY)

    cross_val_groups = get_cross_val_splits(trainX, trainY)

    if args.prop != None : SAVE_PATH = 'models{0}_prop{1}/trial{2}/'.format(args.num_train, args.prop, args.trial)
    else: SAVE_PATH = 'models{0}/trial{1}/'.format(args.num_train, args.trial)

    if args.model_class == 'logreg':
        best_reg, best_reg_tpr, best_reg_fpr = get_best_logreg_reg_via_cross_val(cross_val_groups)
        
        #train and save 
        lrm=LogisticRegression(penalty='l1', C=1.0/2.0**best_reg, verbose=0)
        lrm.fit(trainX,trainY)
        saveto = SAVE_PATH +'lrm.pkl'
        joblib.dump(lrm, saveto)
        print 'SAVING TO ', saveto

        obj = {'tpr': best_reg_tpr, 'fpr': best_reg_fpr}
        json.dump(obj, open(SAVE_PATH+'tpr_fpr.json', 'w')) 

    elif args.model_class == 'loglin':
        best_reg = get_best_loglin_reg_via_cross_val(cross_val_groups, posprior=train_posprior)

        #train and save 
        loglin_wgc = train_loglin(trainX, trainY, best_reg)
        saveto = SAVE_PATH +'loglin.npy'
        np.save(saveto, loglin_wgc)
        print 'SAVING TO ', saveto 

    elif args.model_class == 'mnb':
        best_reg = get_best_mnb_reg_via_cross_val(cross_val_groups, posprior=train_posprior)

        #train and save
        V = trainX.shape[1] 
        w_class_count = get_w_class_count(trainX, trainY)
        mnb_wgc = train_mnb(2.0**best_reg, w_class_count, V)
        saveto = SAVE_PATH +'mnb.npy'
        np.save(saveto, mnb_wgc)
        print 'SAVING TO ', saveto

    else: 
        raise Exception('invalid model class. need to input logreg, loglin, or mnb')



