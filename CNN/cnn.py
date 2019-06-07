import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def build_data():
    """
    Loads data
    """
    files = ["rt-polarity.neg", "rt-polarity.pos"]
    revs = []
    vocab = defaultdict(float)
    max_l = 0
    for i in range(2):
        with open(files[i], "r", encoding="cp1252") as f:
            for line in f:
                orig_rev = clean_str(line.strip())
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":i,
                        "text": orig_rev,
                        "num_words": len(orig_rev.split())}
                if datum["num_words"] > max_l:
                    max_l = datum["num_words"]
                revs.append(datum)
    return revs, vocab, max_l

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())  # 3000000, 300
        binary_len = 4 * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode("latin1")
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = torch.from_numpy(np.frombuffer(f.read(binary_len), dtype='float32'))
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = torch.rand(k) / 2.0 - 0.25

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = torch.zeros([vocab_size+1, k], dtype=torch.float32)
    W[0] = torch.zeros(k, dtype=torch.float32)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    random.shuffle(revs)
    i = 0
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if i < len(revs) / 10:
            test.append(sent)
            i += 1
        else:
            train.append(sent)
    train = torch.tensor(train, dtype=torch.int32)
    test = torch.tensor(test, dtype=torch.int32)
    return train, test

def train_conv_net(datasets,
                   U,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=0.5,
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   sqr_norm_lim=9,
                   non_static=True):
    # Embedding layer
    embedding = None
    return None

def main():
    # read MR dataset
    print("loading data...", end=' ')
    revs, vocab, max_l = build_data()
    print("data loaded!")

    print("number of sentences: " + str(len(revs)))             # 10662
    print("vocab size: " + str(len(vocab)))                     # 18764
    print("max sentence length: " + str(max_l))                 # 56

    # read pre-trained word2vec
    print("loading word2vec vectors...", end=' ')
    w2v = load_bin_vec("GoogleNews-vectors-negative300.bin", vocab)
    print("word2vec loaded!")

    print("num words already in word2vec: " + str(len(w2v)))    # 16448

    W = {}                                                      # torch.Size([18765, 300])
    add_unknown_words(w2v, vocab)
    W["w2v"], word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W["rand"], _ = get_W(rand_vecs)
    print("dataset created!")

    non_static = [True, False, True]
    U = [W["rand"], W["w2v"], W["w2v"]] 
    results = [[],[],[]]
    train, test = make_idx_data(revs, word_idx_map, max_l=max_l, k=300, filter_h=5)    # 9595 1067 X 65
    '''
    for j in range(3):
        perf = train_conv_net(datasets,
                            U[j],
                            lr_decay=0.95,
                            filter_hs=[3,4,5],
                            hidden_units=[100,2],
                            shuffle_batch=True,
                            n_epochs=1, # 25
                            sqr_norm_lim=9,
                            non_static=non_static[j],
                            batch_size=50,
                            dropout_rate=0.5)
        results[j].append(perf)
    '''
    with open("results", "w") as f:
        f.write(str(results))

main()
