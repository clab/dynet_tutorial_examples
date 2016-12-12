from collections import Counter, defaultdict
from itertools import count
import random

import dynet as dy
import numpy as np

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file="CHAR_TRAIN"
test_file="CHAR_DEV"

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def read(fname):
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    with file(fname) as fh:
        for line in fh:
            sent = line.strip().split()
            sent.append("<s>")
            yield sent

train=list(read(train_file))
test=list(read(test_file))
words=[]
wc=Counter()
for sent in train:
    for w in sent:
        words.append(w)
        wc[w]+=1

vw = Vocab.from_corpus([words])
S = vw.w2i["<s>"]

nwords = vw.size()

# DyNet Starts

model = dy.Model()
trainer = dy.AdamTrainer(model)

# Lookup parameters for word embeddings
WORDS_LOOKUP = model.add_lookup_parameters((nwords, 64))

# Word-level LSTM (layers=1, input=64, output=128, model)
RNN = dy.LSTMBuilder(1, 64, 128, model)

# Softmax weights/biases on top of LSTM outputs
W_sm = model.add_parameters((nwords, 128))
b_sm = model.add_parameters(nwords)

# Build the language model graph
def calc_lm_loss(sent):

    dy.renew_cg()
    # parameters -> expressions
    W_exp = dy.parameter(W_sm)
    b_exp = dy.parameter(b_sm)

    # initialize the RNN
    f_init = RNN.initial_state()

    # get the word ids
    wids = [vw.w2i[w] for w in sent]

    # start the rnn by inputting "<s>"
    s = f_init.add_input(WORDS_LOOKUP[wids[-1]]) 

    # feed word vectors into the RNN and predict the next word
    losses = []
    for wid in wids:
        # calculate the softmax and loss
        score = W_exp * s.output() + b_exp
        loss = dy.pickneglogsoftmax(score, wid)
        losses.append(loss)
        # update the state of the RNN
        s = s.add_input(WORDS_LOOKUP[wid]) 
    
    return dy.esum(losses)

num_tagged = cum_loss = 0
for ITER in xrange(50):
    random.shuffle(train)
    for i,s in enumerate(train,1):
        if i % 500 == 0:
            trainer.status()
            print cum_loss / num_tagged
            cum_loss = 0
            num_tagged = 0
        if i % 10000 == 0 or i == len(train)-1:
            dev_loss = dev_words = 0
            for sent in test:
                loss_exp = calc_lm_loss(sent)
                dev_loss += loss_exp.scalar_value()
                dev_words += len(sent)
            print dev_loss / dev_words
        # train on sent

        loss_exp = calc_lm_loss(s)
        cum_loss += loss_exp.scalar_value()
        num_tagged += len(s)
        loss_exp.backward()
        trainer.update()
    print "epoch %r finished" % ITER
    trainer.update_epoch(1.0)


