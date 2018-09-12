from __future__ import division
import argparse 
import json
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize

num_docs = 0
parser = argparse.ArgumentParser()
parser.add_argument("input", help=".json file", type=str)
args = parser.parse_args()

outfile = args.input.split('_')[0]+'/'+args.input.split('/')[1]
ww = open(outfile, 'w')
for line in open(args.input, 'r'):
    obj=json.loads(line)
    #tokenize using nltk 
    toks = [w.lower() for w in word_tokenize(obj['text'])]
    obj["toks"] = toks
    counts = dict((x, toks.count(x)) for x in set(toks))
    obj["counts"] = counts 
    obj["class"] = int(obj["stars"] >= 3)
    num_docs +=1
    json.dump(obj, ww)
    ww.write('\n')
ww.close()
print 'wrote ', outfile

print 'NUM OF DOCS = ', num_docs




