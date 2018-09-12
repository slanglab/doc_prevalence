from __future__ import division
import numpy as np 
from sklearn.feature_extraction import DictVectorizer

def get_vocab(traindicts):
    #what's my vocab 
    dv=DictVectorizer()
    _trainX=dv.fit_transform(traindicts)
    _dv_vocab=np.array(dv.feature_names_)

    # Prune the vocab
    # this keeps all words that are in >5 docs 
    xx=_trainX.copy()
    xx[xx>0]=1
    w_df = np.asarray(xx.sum(0)).flatten()
    new_vocab_mask = w_df >= 5
    print "Orig vocab %d, pruned %d" % (len(w_df), np.sum(new_vocab_mask))
    trainX = _trainX[:,new_vocab_mask]
    vocab = _dv_vocab[new_vocab_mask]
    word2num = {w:i for (i,w) in enumerate(vocab)}
    return trainX, word2num

