from __future__ import division
import numpy as np
from scipy import io, sparse
from scipy.misc import logsumexp
import sys,os,re,math,scipy,glob,argparse
import json as ujson 
import cPickle as pickle
from collections import defaultdict,Counter
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import subprocess, json
import utils

parser = argparse.ArgumentParser()
parser.add_argument("num_train", help="number of training documents", type=int)
parser.add_argument("--trial", help="trial for ntrain2000", type=int, default=None)
args = parser.parse_args()

traindicts = []
trainY = []
class2wordcount = defaultdict(list)
train_text_seq = [] #for the LSTM 

PATH='train'+str(args.num_train)
file_to_open = PATH+'/'+PATH+'.json'

if args.trial != None:
    PATH += '/trial'+str(args.trial)
    file_to_open = PATH + '/train.json'

#READ IN TRAINING DATA
for line in open(file_to_open):
    dd = ujson.loads(line)
    counts = dd['counts'].copy()
    traindicts.append(counts)
    cc = dd['class']
    trainY.append(cc)
    class2wordcount[cc].append(sum(dd['counts'].values()))
trainY = np.array(trainY)
trainX, word2num = utils.get_vocab(traindicts)

#save these so they can be used later on 
PATH += '/'
scipy.sparse.save_npz(PATH+'trainX', trainX)
np.save(PATH+'trainY', trainY)
w1 = open(PATH+'word2num.json', 'w')
json.dump(word2num, w1)

