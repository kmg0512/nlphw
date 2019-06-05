import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def build_data_cv(cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = "rt-polarity.pos"
    neg_file = "rt-polarity.neg"
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(str(line).strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(str(line).strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
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
    return string.strip() if TREC else string.strip().lower()

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = torch.zeros(1).element_size() * layer1_size
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
               word_vecs[word] = torch.from_numpy(np.fromstring(f.read(binary_len), dtype='float32'))
            else:
                f.read(binary_len)
            printProgress(line, vocab_size, 'Progress:', 'Complete', 1, 50)
    return word_vecs

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = torch.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = torch.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = torch.random.uniform(-0.25,0.25,k)

def main():
    # read MR dataset
    revs, vocab = build_data_cv(cv=10, clean_string=True)
    with open("rev", "w") as f:
        f.write(str(revs))
    with open("vocab", "w") as f:
        f.write(str(vocab))

    # read pre-trained word2vec
    w2v = load_bin_vec("GoogleNews-vectors-negative300.bin", vocab)
    with open("w2v", "w") as f:
        f.write(str(w2v))

    W = {}
    #W["w2v"], word_idx_map = get_W(w2v, k=300)

    rand_vecs = {}
    #add_unknown_words(rand_vecs, vocab)
    #W["rand"], _ = get_W(rand_vecs, k=300)

    #U = [W["rand"], W["w2v"], W["w2v"]]
    non_static = [True, False, True]
    results = [[],[],[]]

main()
