from __future__ import division
from collections import defaultdict
import ujson as json
import numpy as np 
import random 
random.seed(1234)
np.random.seed(1234)

id2all = defaultdict(list)

num_docs = 0 
with open('yelp_dataset_challenge_round9/yelp_academic_dataset_review.json','r') as r:
    for line in r: 
        obj = json.loads(line)
        id2all[obj['business_id']].append(obj)
        num_docs +=1 
print 'READ {0} DOCS'.format(num_docs)

possible_test = []
postest_weights = []

MIN_NUM_RVW = 200
NUM_TEST_BUS = 500

for busid, objlist in id2all.iteritems():
    ndocs = len(objlist)
    if  ndocs >= MIN_NUM_RVW: 
        possible_test.append(busid)
        postest_weights.append(ndocs)
print 'NUM BUSINESSES >{0} REVIEWS = {1}'.format(MIN_NUM_RVW, len(possible_test))

#we need a probability vector of all the viable possible test businesses 
#p_i = (# docs for bus i )/ (total num docs)
postest_weights = np.array(postest_weights)
postest_weights = postest_weights / np.sum(postest_weights)

assert len(postest_weights) == len(possible_test)

#now get the test businesses
nottrain = np.random.choice(possible_test, NUM_TEST_BUS*2, p=postest_weights, replace=False)
test = nottrain[0:NUM_TEST_BUS]
dev = nottrain[NUM_TEST_BUS:]

assert len(test) == NUM_TEST_BUS
assert len(dev) == NUM_TEST_BUS
print 'NUM TEST SET = {0}'.format(len(test))

items=id2all.items()
for busid, objlist in items:
    if busid in test: 
        w = open('test_nopreproc/'+str(busid), 'w')
        for obj in objlist:
            json.dump(obj, w)
            w.write('\n')
        w.close()
    elif busid in dev: 
        w = open('dev_nopreproc/'+str(busid), 'w')
        for obj in objlist:
            json.dump(obj, w)
            w.write('\n')
        w.close()
    else: 
        w = open('train_nopreproc/'+str(busid), 'w')
        for obj in objlist:
            json.dump(obj, w)
            w.write('\n')
        w.close()
