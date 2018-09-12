from __future__ import division
import json 
import numpy as np 
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

def print_metrics(eval_metrics, model_type, reg_strength, num_train, trainprop, trial):
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

    ss = '%s,'*5 + '%.5f,'*14
    print ss % (trial, model_type, reg_strength, num_train, trainprop, mae_map, mse_map, mae_mean, mse_mean, m_var, m_bias_map, m_bias_mean, ci_5, ci_9, ci_width5, ci_width9, m_post_exp, error_forecast_bias, mae_mae)

def abline(slope, intercept, plt):
    """Plot a line from slope and intercept"""  # https://stackoverflow.com/questions/7941226/add-line-based-on-slope-and-intercept-in-matplotlib
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-', color='black')

def get_median(data_dict):
    ss = sorted(data_dict.items(), key=lambda (k, v): v)
    median_idx = (len(ss)//2)-1
    return ss[median_idx]

def get_median_eval_metrics(models_list, prop=False):
    if prop: LOAD_PATH = '../code/eval/results_prop1/'
    else: LOAD_PATH = '../code/eval/results/'

    model2triall2mae = defaultdict(lambda: defaultdict(int))
    model2triall2eval_metrics = defaultdict(lambda: defaultdict(dict))

    for model in models_list:
        for trial in np.arange(1, 11, 1):
            if model == 'readme':
                if prop: readme_file = '../code/readme_our_experiments/save_eval/prop1/trial{0}.json'.format(trial)
                else: readme_file = '../code/readme_our_experiments/save_eval/nat/trial{0}.json'.format(trial) 
                eval_metrics = json.load(open(readme_file, 'r'))

            else: 
                eval_metrics  = json.load(open(LOAD_PATH+'{0}_trial{1}.json'.format(model, trial), 'r')) 
            
            mae = np.mean(np.array(eval_metrics['ae_map']))
            model2triall2mae[model][trial] = mae
            model2triall2eval_metrics[model][trial] = eval_metrics

    model2median_trial = {}
    for model in models_list:
        median_trial, median_mae = get_median(model2triall2mae[model])
        print '{0:8}, median_trial={1:2}, median_mae={2}'.format(model, median_trial, median_mae)
        model2median_trial[model] = median_trial

    model2graph_eval = {}
    for model, median_trial in model2median_trial.iteritems():
        model2graph_eval[model] = model2triall2eval_metrics[model][median_trial]

    return model2graph_eval

def graph_mae_plain(model2graph_eval):
    #w/o confidence intervals
    plt.figure(figsize=(12,12))
    for i, model in enumerate(model2graph_eval):
        graph_eval = model2graph_eval[model]
        plt.subplot(3,3,i+1)
        plt.scatter(graph_eval['true_labelmeans'],graph_eval['map_est'])
        plt.xlabel('test gold proportion')
        plt.ylabel('predicted MAP proportion')
        plt.title(model)
        plt.xlim(0,1);plt.ylim(0,1)
        abline(1,0, plt)
    plt.tight_layout()
    return plt

def graph_mae_ci(model2graph_eval):
    #w/ confidence intervals 
    plt.figure(figsize=(12,12))
    for i, model in enumerate(model2graph_eval):
        graph_eval = model2graph_eval[model]
        plt.subplot(3,3,i+1)
        #plt.scatter(graph_eval['true_labelmeans'],graph_eval['map_est'])
        
        for jj in xrange(500):
            ci_width = graph_eval['conf_width0.9'][jj]
            map_est = graph_eval['map_est'][jj]
            true_theta = graph_eval['true_labelmeans'][jj]
            ci_coverage = graph_eval['conf_0.9'][jj]
            
            if ci_coverage == 0: color = 'green'
            elif ci_coverage == 1: color = 'blue'
            #ci line 
            plt.plot((true_theta, true_theta), (map_est- ci_width/2.0, map_est+ ci_width/2.0), 'k-', alpha=0.2)
            
            plt.scatter(true_theta, map_est, color=color) 
        
        plt.xlabel('test gold proportion')
        plt.ylabel('predicted MAP proportion')
        plt.title(model)
        plt.xlim(0,1);plt.ylim(0,1)
        abline(1,0, plt)
    plt.tight_layout()
    return plt

def graph_log_testsize_v_diff(model2graph_eval):
    """
    test group size, vs. ypred-ygold
    """
    plt.figure(figsize=(12,12))
    for i, model in enumerate(model2graph_eval):
        graph_eval = model2graph_eval[model]
        plt.subplot(3,3,i+1)
        signed_difference = np.array(graph_eval['map_est'])-np.array(graph_eval['true_labelmeans']) #ypred-ygold
        plt.scatter(np.log(np.array(graph_eval['ndocs'])), signed_difference)
        plt.xlabel('LOG num. docs in test group')
        plt.ylabel('ypred-ygold')
        plt.title(model)
        plt.axhline(y=0, color='k')
        plt.ylim(-.7, .4) 
    plt.tight_layout()
    return plt

def get_bin_num(ndocs, split2ndocs):
    for bin_num, ndoc_range in split2ndocs.iteritems():
        if ndocs >= ndoc_range[0] and ndocs < ndoc_range[1]:
            return bin_num

def get_bin_size(model2graph_eval, nbins=3):
    #get bin sizes 
    graph_eval = model2graph_eval['mnb']
    sorted_ndocs = sorted(graph_eval['ndocs'])
    split = 500//nbins
    split2ndocs = {}
    end = 0
    for ii in xrange(3):
        start = end
        end = start + split
        split2ndocs[ii] = (sorted_ndocs[start], sorted_ndocs[end])

    #adjust the last bin 
    split2ndocs[2] = (635, 10000)
    print split2ndocs
    return split2ndocs

def get_binned_CI_rate(model2graph_eval, model, split2ndocs):
    graph_eval = model2graph_eval[model]
    bin2CI_rates_lists = defaultdict(list)
    for testgroup_idx in xrange(len(graph_eval['ndocs'])):
        ndocs = graph_eval['ndocs'][testgroup_idx]
        bin_num = get_bin_num(ndocs, split2ndocs)
        bin2CI_rates_lists[bin_num].append(graph_eval['conf_0.9'][testgroup_idx])
    bin2CI_rates = {}
    for bin_num, CI_rates_lists in bin2CI_rates_lists.iteritems():
        rt = np.mean(np.array(CI_rates_lists))
        bin2CI_rates[bin_num] = rt
    return bin2CI_rates

def get_binned_CI_width(model2graph_eval, model, split2ndocs):
    graph_eval = model2graph_eval[model]
    bin2CI_widths_lists = defaultdict(list)
    for testgroup_idx in xrange(len(graph_eval['ndocs'])):
        ndocs = graph_eval['ndocs'][testgroup_idx]
        bin_num = get_bin_num(ndocs, split2ndocs)
        bin2CI_widths_lists[bin_num].append(graph_eval['conf_width0.9'][testgroup_idx])
    bin2CI_widths = {}
    for bin_num, CI_widths_lists in bin2CI_widths_lists.iteritems():
        mean_width = np.mean(np.array(CI_widths_lists))
        bin2CI_widths[bin_num] = mean_width
    return bin2CI_widths

def plot_binned_CI_rates(model2graph_eval, split2ndocs):
    plt.figure(figsize=(12,12))
    for i, model in enumerate(model2graph_eval):
        plt.subplot(3,3,i+1)
        toplot = get_binned_CI_rate(model2graph_eval, model, split2ndocs)
        print model, toplot
        plt.bar(toplot.keys(), toplot.values(), align='center')
        plt.xticks(toplot.keys(), toplot.keys())
        plt.xlabel('quantile, binned by number of docs in test group')
        plt.ylabel('CI coverage rate')
        plt.ylim(0, 0.7)
        plt.title(model)
    plt.tight_layout()
    return plt

def plot_binned_CI_widths(model2graph_eval, split2ndocs):
    plt.figure(figsize=(12,12))
    for i, model in enumerate(model2graph_eval):
        plt.subplot(3,3,i+1)
        toplot = get_binned_CI_width(model2graph_eval, model, split2ndocs)
        print model, toplot
        plt.bar(toplot.keys(), toplot.values(), align='center')
        plt.xticks(toplot.keys(), toplot.keys())
        plt.xlabel('quantile, binned by number of docs in test group')
        plt.ylabel('average CI width')
        plt.ylim(0, 0.12)
        plt.title(model)
    plt.tight_layout()
    return plt








