from __future__ import division
import json 
import numpy as np
from collections import defaultdict

def get_mae(eval_metrics):
    return np.mean(np.array(eval_metrics['ae_map']))

def get_ci(eval_metrics):
    return np.mean(np.array(eval_metrics['conf_0.9']))

if __name__ == "__main__":
    model2tsize2mae = defaultdict(lambda : defaultdict(list))
    model2tsize2ci = defaultdict(lambda : defaultdict(list))
    for model in ['pcc', 'hybrid']:
        for tsize in np.arange(7, 14, 1):
            for trial in np.arange(1, 11, 1):
                eval_metrics  = json.load(open('results_tsize{0}/{1}_trial{2}.json'.format(tsize, model, trial), 'r'))
                mae = get_mae(eval_metrics)
                ci = get_ci(eval_metrics)
                model2tsize2mae[model][tsize].append(mae)
                model2tsize2ci[model][tsize].append(ci)

    json.dump(model2tsize2mae, open('trainsize_forgraphs/model2tsize2mae.json', 'w'))
    json.dump(model2tsize2ci, open('trainsize_forgraphs/model2tsize2ci.json', 'w'))