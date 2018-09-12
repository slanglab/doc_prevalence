from __future__ import division
import json, argparse 
import numpy as np
from random import shuffle
import utils
from collections import defaultdict
import scipy

parser = argparse.ArgumentParser()
parser.add_argument("prop", help="proportion/prevalence you want to select for, 1 thru 9 for 0.1 thru 0.9", type=int)
parser.add_argument("trial", help="trial for random selection of the training docs", type=int)
args = parser.parse_args()
NUMTRAIN = 2000
trial = args.trial 
prop = args.prop
prop_value = prop/10.0 

#(1) get the right training proportions
#read in and split into pos and neg
PATH='train{0}_prop{1}/trial{2}/'.format(NUMTRAIN, prop, trial)
pos = []
neg = []
for line in open(PATH+'train_toselect.json'):
    dd = json.loads(line)
    if dd['class'] == 0: 
        neg.append(dd)
    elif dd['class'] == 1:
        pos.append(dd)
    else: raise Exception('no class!')
shuffle(pos)
shuffle(neg)

num_pos_to_sample = int(NUMTRAIN*prop_value )
num_neg_to_sample = NUMTRAIN - num_pos_to_sample
selected_docs = pos[0:num_pos_to_sample] + neg[0:num_neg_to_sample]
assert len(selected_docs) == NUMTRAIN

#(2) get the vocab for that training proportions 
traindicts = []
trainY = []
class2wordcount = defaultdict(list)
for dd in selected_docs:
    traindicts.append(dd['counts'].copy())
    cc = dd['class']
    trainY.append(cc)
    class2wordcount[cc].append(sum(dd['counts'].values()))
trainY = np.array(trainY)

trainX, word2num = utils.get_vocab(traindicts)

#save these
scipy.sparse.save_npz(PATH+'trainX', trainX)
np.save(PATH+'trainY', trainY)
w1 = open(PATH+'word2num.json', 'w')
json.dump(word2num, w1)