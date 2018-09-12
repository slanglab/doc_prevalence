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

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'utils'))
import infer, ecdf

def load_model_trials(model_type, num_train, trial, prop=None):
    if prop != None: LOAD_PATH = '../train_all/models{0}_prop{1}/trial{2}/'.format(num_train, prop, trial)
    else: LOAD_PATH = '../train_all/models{0}/trial{1}/'.format(num_train, trial)

    if model_type in ['cc', 'acc', 'pcc', 'hybrid']:
        return joblib.load( LOAD_PATH+'lrm.pkl')

    elif model_type == 'mnb':
        return np.load(LOAD_PATH+'mnb.npy')

    elif model_type == 'loglin':
        return np.load(LOAD_PATH+'loglin.npy')

    raise Exception('invalid model input')

def do_acc(p0, tpr, fpr):
    """
    adjusted classify and count method 

    \hat{p} - (p0 - fpr)/(tpr - fpr)
    """
    if tpr == fpr == 1.0: 
        raise Exception('tpr and fpr both equal to 1.0')

    p_hat = (p0 - fpr) / (tpr - fpr)

    #clip p_hat
    if p_hat < 1e-15: p_hat = 1e-15
    if p_hat > (1-1e-15): p_hat = (1-1e-15) 
    return p_hat

def get_training_prevelance(trial):
    #TODO: go through the training docs and get the true prevelance 
    if args.prop != None: 
        LOAD_TRAIN_PATH = '/home/kkeith/docprop/yelp_data/train2000_prop{0}/trial{1}/'.format(args.prop, trial)
    else:
        LOAD_TRAIN_PATH = '/home/kkeith/docprop/yelp_data/train2000/trial{0}/'.format(trial) 

    trainY = np.load(LOAD_TRAIN_PATH+'trainY.npy')
    return np.mean(trainY)

def get_tpr_fpr(num_train, trial, prop=False):
    if prop: LOAD_PATH = '../train_all/models{0}_prop1/trial{1}/'.format(num_train, trial)
    else: LOAD_PATH = '../train_all/models{0}/trial{1}/'.format(num_train, trial)
    trained_rates = json.load(open(LOAD_PATH+'tpr_fpr.json', 'r'))
    tpr, fpr = trained_rates['tpr'], trained_rates['fpr']
    return tpr, fpr 

#discriminative methods
def cc_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type):
    model= load_model_trials(model_type, num_train, trial, prop=prop)
    map_est = np.mean(model.predict(tx))
    eval_metrics['map_est'].append(map_est)
    ae_map = np.abs(map_est - ymean_true)
    eval_metrics['ae_map'].append(ae_map)
    return eval_metrics

def acc_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type):
    model= load_model_trials(model_type, num_train, trial, prop=prop)
    tpr, fpr = get_tpr_fpr(num_train, trial, prop=prop)
    cc_est = np.mean(model.predict(tx))
    map_est = do_acc(cc_est, tpr, fpr)
    eval_metrics['map_est'].append(map_est)
    ae_map = np.abs(map_est - ymean_true)
    eval_metrics['ae_map'].append(ae_map)
    return eval_metrics

#generative methods 
def pcc_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type):
    model= load_model_trials(model_type, num_train, trial, prop=prop)
    py = model.predict_proba(tx)[:,1]
    map_est = np.mean(py)
    eval_metrics['map_est'].append(map_est)
    log_post_probs = get_lr_samples(py)

    catdist = ecdf.CategDist(log_post_probs)
    eval_metrics = get_extra_generative_metrics(eval_metrics, catdist, map_est, ymean_true)
    eval_metrics = get_posterior_expected_mae(eval_metrics, catdist, map_est)
    eval_metrics = get_ci_coverage_and_widths(eval_metrics, catdist, ymean_true)
    return eval_metrics

def mnb_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type):
    return same_mnb_loglin(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)

def loglin_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type):
    return same_mnb_loglin(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)

def same_mnb_loglin(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type):
    """
    loglin and mnb are the same except that they load different language models (captured in the model, type variable)
    """
    model= load_model_trials(model_type, num_train, trial, prop=prop)
    log_post_probs = infer.mnb_mll_curve(tx, model)
    return after_log_post_probs(eval_metrics, log_post_probs)

def hybrid_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type):
    model= load_model_trials(model_type, num_train, trial, prop=prop)
    training_prevelance = get_training_prevelance(trial)
    logodds = model.decision_function(tx)
    log_post_probs = infer.hybrid_mll_curve_stable(logodds, training_prevelance)
    return after_log_post_probs(eval_metrics, log_post_probs)

def after_log_post_probs(eval_metrics, log_post_probs):
    log_prior = get_beta_prior()
    log_post_probs = np.add(log_post_probs, log_prior)
    map_est = generative_get_map_est(log_post_probs)
    eval_metrics['map_est'].append(map_est)
    value_to_unorm_logprob = {t: ll for t, ll in zip(infer.trange, log_post_probs)}
    catdist = ecdf.dist_from_logprobs(value_to_unorm_logprob)
    eval_metrics = get_extra_generative_metrics(eval_metrics, catdist, map_est, ymean_true)
    eval_metrics = get_posterior_expected_mae(eval_metrics, catdist, map_est)
    eval_metrics = get_ci_coverage_and_widths(eval_metrics, catdist, ymean_true)
    return eval_metrics

def get_extra_generative_metrics(eval_metrics, catdist, map_est, ymean_true):
    mean_est = catdist.post_mean #posterior mean 
    eval_metrics['mean_est'].append(mean_est)
    eval_metrics['post_var'].append(catdist.post_var)
    ae_map = np.abs(map_est - ymean_true)
    eval_metrics['ae_map'].append(ae_map)
    eval_metrics['ae_mean'].append(np.abs(mean_est - ymean_true))
    return eval_metrics

def get_ci_coverage_and_widths(eval_metrics, catdist, ymean_true):
    for conf_level in [0.5, 0.9]:
        conf_intvl = catdist.hdi(conf_level)
        if conf_intvl[0] <= ymean_true and ymean_true <= conf_intvl[1]:
            eval_metrics['conf_'+str(conf_level)].append(1)
        else: 
            eval_metrics['conf_'+str(conf_level)].append(0)

        width = conf_intvl[1] - conf_intvl[0]
        eval_metrics['conf_width'+str(conf_level)].append(width)
    return eval_metrics

def get_posterior_expected_mae(eval_metrics, catdist, map_est):
    post_exp_ae = 0 
    for val, prob in catdist.value_to_prob.iteritems():
        post_exp_ae += prob * np.abs(map_est - val)
    eval_metrics['post_exp_ae'].append(post_exp_ae) 
    return eval_metrics

if __name__ == "__main__":
    """
    outputs "eval_metrics" dictionary for the given model type and trial 

    next you'll want to run analysis.py from this same directory 

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", help="model type, choose from (cc, acc, pcc, loglin, mnb, hybrid)", type=str)
    parser.add_argument("trial", help="training resampling, 1 thru 10", type=int)
    parser.add_argument("--prop", help='training prevalence', type=int, default=None)
    parser.add_argument("--ntrain", help='training prevalence', type=int, default=None)
    args = parser.parse_args()

    if args.ntrain != None: num_train = args.ntrain 
    else: num_train = 2000

    model_type = args.model_type
    trial = args.trial 
    prop = args.prop 

    #load test data 
    if args.prop != None: TEST_FILES = 'saved_testdata/trial{0}_prop{1}.joblib'.format(args.trial, args.prop)
    else: 
        if args.ntrain != None: TEST_FILES = 'saved_testdata/trial{0}_ntrain{1}.joblib'.format(args.trial, args.ntrain)
        else: TEST_FILES = 'saved_testdata/trial{0}.joblib'.format(args.trial)
    loaded_testdata = joblib.load(TEST_FILES)

    #evaluate 
    eval_metrics = defaultdict(list)
    for tx, ty in zip(loaded_testdata['all_xs'], loaded_testdata['all_ygold']):
        eval_metrics['ndocs'].append(len(ty))
        ymean_true = np.mean(ty)
        eval_metrics['true_labelmeans'].append(ymean_true)

        #discriminative models
        if model_type == 'cc': eval_metrics = cc_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)
        elif model_type == 'acc': eval_metrics = acc_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)

        #generative models 
        elif model_type == 'pcc': eval_metrics = pcc_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)
        elif model_type == 'mnb': eval_metrics = mnb_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)
        elif model_type == 'loglin': eval_metrics = loglin_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)
        elif model_type == 'hybrid': eval_metrics = hybrid_method(eval_metrics, tx, ymean_true, num_train, trial, prop, model_type)
        else: raise Exception('Invalid model type! Model type must be one of (cc, acc, pcc, mnb, loglin, hybrid)')

    sys.stderr.write("\n")

    #save eval metrics 
    if args.prop != None: OUT_PATH = 'results_prop{0}/{1}_trial{2}.json'.format(args.prop, model_type, trial)
    else: 
        if args.ntrain != None: OUT_PATH = 'results_tsize{0}/{1}_trial{2}.json'.format(args.ntrain, model_type, trial)
        else: OUT_PATH = 'results/{0}_trial{1}.json'.format(model_type, trial)
    ww = open(OUT_PATH, 'w')
    json.dump(eval_metrics, ww)
    print 'saved to ', OUT_PATH
    ww.close()
