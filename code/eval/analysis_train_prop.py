from __future__ import division
import json 
import numpy as np
from collections import defaultdict

def ave_trials_print_metrics(eval_metrics_list, model_type, num_train, trainprop):
    overall = defaultdict(list)
    for eval_metrics in eval_metrics_list: 
        for k, v in eval_metrics.iteritems():
            eval_metrics[k] = np.array(v)

        overall['mae_map'].append(np.mean(eval_metrics['ae_map']))
        overall['mse_map'].append(np.mean(np.square(eval_metrics['map_est']- eval_metrics['true_labelmeans'])))
        overall['mae_mean'].append(np.mean(eval_metrics['ae_mean']))
        overall['mse_mean'].append(np.mean(np.square(eval_metrics['mean_est']- eval_metrics['true_labelmeans'])))

        overall['m_var'].append(np.mean(eval_metrics['post_var']))
        overall['m_bias_map'].append(np.mean(eval_metrics['map_est']- eval_metrics['true_labelmeans']))
        overall['m_bias_mean'].append(np.mean(eval_metrics['mean_est']- eval_metrics['true_labelmeans']))

        overall['ci_5'].append(np.mean(eval_metrics['conf_0.5']))
        overall['ci_9'].append(np.mean(eval_metrics['conf_0.9']))

        overall['ci_width5'].append(np.mean(eval_metrics['conf_width0.5']))
        overall['ci_width9'].append(np.mean(eval_metrics['conf_width0.9']))

        overall['m_post_exp'].append(np.mean(eval_metrics['post_exp_ae']))
        overall['error_forecast_bias'].append(np.mean(eval_metrics['post_exp_ae'] - eval_metrics['ae_map']))

        overall['mae_mae'].append(np.mean(np.abs(eval_metrics['post_exp_ae'] - eval_metrics['ae_map'])))

    toprint = {}
    for metric, ll in overall.iteritems():
        toprint[metric] = np.mean(np.array(ll))

    ss = '%s,'*3 + '%.5f,'*14
    print ss % (model_type, num_train, trainprop, toprint['mae_map'], toprint['mse_map'], toprint['mae_mean'], toprint['mse_mean'], toprint['m_var'], 
                toprint['m_bias_map'], toprint['m_bias_mean'], toprint['ci_5'], toprint['ci_9'], toprint['ci_width5'], toprint['ci_width9'], 
                toprint['m_post_exp'], toprint['error_forecast_bias'], toprint['mae_mae'])

    return overall['mae_map'], overall['ci_9']

def ave_trials_print_metrics_cc_acc(eval_metrics_list, model_type, num_train, trainprop):
    overall_metrics = defaultdict(list)

    for eval_metrics in eval_metrics_list:
        for k, v in eval_metrics.iteritems():
            eval_metrics[k] = np.array(v)

        overall_metrics['mae_map'].append(np.mean(eval_metrics['ae_map']))
        overall_metrics['m_bias_map'].append(np.mean(eval_metrics['map_est']- eval_metrics['true_labelmeans']))

    toprint = {}
    for metric, ll in overall_metrics.iteritems():
        toprint[metric] = np.mean(np.array(ll))

    ss = '%s,'*3 + '%.5f,'*2
    print ss % (model_type, num_train, trainprop, toprint['mae_map'], toprint['m_bias_map'])
    return overall_metrics['mae_map']

def get_mae(eval_metrics):
    return np.mean(np.array(eval_metrics['ae_map']))

def get_ci(eval_metrics):
    return np.mean(np.array(eval_metrics['conf_0.9']))

if __name__ == "__main__":

    model2prop2mae = defaultdict(lambda : defaultdict(list))
    model2prop2ci = defaultdict(lambda : defaultdict(list))

    for model in ['pcc', 'hybrid']:
        for prop in np.arange(1, 10, 1):
            for trial in np.arange(1, 11, 1):
                eval_metrics  = json.load(open('results_prop{0}/{1}_trial{2}.json'.format(prop, model, trial), 'r'))
                mae = get_mae(eval_metrics)
                ci = get_ci(eval_metrics)
                model2prop2mae[model][prop].append(mae)
                model2prop2ci[model][prop].append(ci)

    json.dump(model2prop2mae, open('trainprop_forgraphs/model2prop2mae.json', 'w'))
    json.dump(model2prop2ci, open('trainprop_forgraphs/model2prop2ci', 'w'))

    for model in ['acc', 'cc']:
        for prop in np.arange(1, 10, 1):
            eval_metrics_list = []
            for trial in np.arange(1, 11, 1):
                eval_metrics  = json.load(open('results_prop{0}/{1}_trial{2}.json'.format(prop, model, trial), 'r'))
                eval_metrics_list.append(eval_metrics)
            mae = ave_trials_print_metrics_cc_acc(eval_metrics_list, model, 2000, prop/10.0) 
            model2mae[model].append(mae)

    for model in ['pcc', 'mnb', 'loglin', 'hybrid']:
    for model in ['pcc', 'hybrid']:
        for prop in np.arange(1, 10, 1):
            eval_metrics_list = []
            for trial in np.arange(1, 11, 1):
                eval_metrics  = json.load(open('results_prop{0}/{1}_trial{2}.json'.format(prop, model, trial), 'r'))
                eval_metrics_list.append(eval_metrics)
            mae, ci = ave_trials_print_metrics(eval_metrics_list, model, 2000, prop/10.0)
            model2mae[model].append(mae)
            model2ci[model].append(ci)

    json.dump(model2mae, open('trainprop_forgraphs/model2mae.json', 'w'))
    json.dump(model2ci, open('trainprop_forgraphs/model2ci.json', 'w'))
