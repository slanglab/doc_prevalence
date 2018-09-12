from __future__ import division
import json 
from collections import defaultdict 
import numpy as np 

"""
Saves for graphing: 
    eval_metrics, dict
        keys: 'ae_map', 'bias', 'map_est', 'true_labelmeans'
        values: lists corresponding to the test groups  

"""

def get_pertrial_eval_metrics(setting, trial):
    eval_metrics_per_trial = defaultdict(list)
    for group in xrange(1, 501):
        for idx, line in enumerate(open('results/{s}/trial{t}/group{g}.csv'.format(s=setting, t=trial, g=group), 'r')):
            eval_metrics_per_trial[idx2label[idx]].append(float(line.strip())) 
    fout = 'save_eval/{s}/trial{t}.json'.format(s=setting, t=trial)
    
    mae = np.mean(np.array(eval_metrics_per_trial['ae_map']))
    bias = np.mean(np.array(eval_metrics_per_trial['bias']))
    print '\tTRIAL={t}, MAE={m}, BIAS={b}'.format(t=trial, m=mae, b=bias)
    json.dump(eval_metrics_per_trial, open(fout, 'w'))
    return mae, bias 

idx2label = {0: 'ae_map', 1:'bias', 2:'map_est', 3:'true_labelmeans'}

for setting in ['nat', 'prop1']:
    print '========='
    print 'SETTING=', setting
    mae_alltrials = []; bias_alltrials = [] 
    for trial in xrange(1, 11):
        mae, bias = get_pertrial_eval_metrics(setting, trial)
        mae_alltrials.append(mae)
        bias_alltrials.append(bias)
    print 'MAE (across trials)=', np.mean(np.array(mae_alltrials))
    print 'BIAS (across trials)=', np.mean(np.array(bias_alltrials))
    print '========='

