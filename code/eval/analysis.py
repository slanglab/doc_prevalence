from __future__ import division
import json 
import numpy as np
from collections import defaultdict

def print_metrics(eval_metrics, model_type, num_train, trainprop, trial):
    for k, v in eval_metrics.iteritems():
        eval_metrics[k] = np.array(v)

    mae_map = np.mean(eval_metrics['ae_map'])
    mse_map = np.mean(np.square(eval_metrics['map_est']- eval_metrics['true_labelmeans']))
    mae_mean = np.mean(eval_metrics['ae_mean'])
    mse_mean = np.mean(np.square(eval_metrics['mean_est']- eval_metrics['true_labelmeans']))

    m_var = np.mean(eval_metrics['post_var'])
    m_bias_map = np.mean(eval_metrics['map_est']- eval_metrics['true_labelmeans'])
    m_bias_mean = np.mean(eval_metrics['mean_est']- eval_metrics['true_labelmeans'])

    ci_5 = np.mean(eval_metrics['conf_0.5'])
    ci_9 = np.mean(eval_metrics['conf_0.9'])

    ci_width5 = np.mean(eval_metrics['conf_width0.5'])
    ci_width9 = np.mean(eval_metrics['conf_width0.9'])

    m_post_exp = np.mean(eval_metrics['post_exp_ae'])
    error_forecast_bias = np.mean(eval_metrics['post_exp_ae'] - eval_metrics['ae_map'])

    mae_mae = np.mean(np.abs(eval_metrics['post_exp_ae'] - eval_metrics['ae_map']))

    ss = '%s,'*4 + '%.5f,'*14
    print ss % (trial, model_type, num_train, trainprop, mae_map, mse_map, mae_mean, mse_mean, m_var, m_bias_map, m_bias_mean, ci_5, ci_9, ci_width5, ci_width9, m_post_exp, error_forecast_bias, mae_mae)

def print_metrics_cc_acc(eval_metrics, model_type, num_train, trainprop, trial):
    for k, v in eval_metrics.iteritems():
        eval_metrics[k] = np.array(v)

    mae_map = np.mean(eval_metrics['ae_map'])
    m_bias_map = np.mean(eval_metrics['map_est']- eval_metrics['true_labelmeans'])
    ss = '%s,'*4 + '%.5f,'*2
    print ss % (trial, model_type, num_train, trainprop, mae_map, m_bias_map)

for model in ['cc', 'acc']:
    for trial in np.arange(1, 11, 1):
        eval_metrics  = json.load(open('results/{0}_trial{1}.json'.format(model, trial), 'r'))
        print_metrics_cc_acc(eval_metrics, model, 2000, 'natural', trial) 
    print 

for model in ['cc', 'acc']:
    for trial in np.arange(1, 11, 1):
        eval_metrics  = json.load(open('results_prop1/{0}_trial{1}.json'.format(model, trial), 'r'))
        print_metrics_cc_acc(eval_metrics, model, 2000, 0.1, trial) 
    print 

for model in ['pcc', 'mnb', 'loglin', 'hybrid']:
    for trial in np.arange(1, 11, 1):
        eval_metrics  = json.load(open('results/{0}_trial{1}.json'.format(model, trial), 'r'))
        print_metrics(eval_metrics, model, 2000, 'natural', trial) 
    print 

for model in ['pcc', 'mnb', 'loglin', 'hybrid']:
    for trial in np.arange(1, 11, 1):
        eval_metrics  = json.load(open('results_prop1/{0}_trial{1}.json'.format(model, trial), 'r'))
        print_metrics(eval_metrics, model, 2000, 0.1, trial) 
    print 