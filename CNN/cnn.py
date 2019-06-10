import random
import re
import sys
import time
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


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
                datum  = {  "y":i,
                            "text": orig_rev}
                if len(orig_rev.split()) > max_l:
                    max_l = len(orig_rev.split())
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
    return word_vecs, layer1_size

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l=56, k=300):
    """
    Transforms sentences into a 2-d matrix.
    """
    train_x_idx, train_y, test_x_idx, test_y = [], [], [], []
    random.shuffle(revs)
    for rev, i in zip(revs, range(len(revs))):
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k)
        if i < len(revs) / 10:
            test_x_idx.append(sent)
            test_y.append(rev["y"])
        else:
            train_x_idx.append(sent)
            train_y.append(rev["y"])
    train_y = torch.Tensor(train_y)
    test_y = torch.Tensor(test_y)
    return train_x_idx, train_y, test_x_idx, test_y

def make_data(x_idx, W, max_l=56, k=300):
    x = [reduce(lambda x,y:x+y, [W[idx] for idx in [sent for sent in x_idx]])]
    print(x)
    return None

# def train_conv_net(train_x, train_y, W, non_static, h=[3,4,5], feature=100, p=0.5, s=3, batch=50, k=300):
#     x = W[i] for i in train_x
#     #x = torch.tensor(W[i] for i in train_x)
#     print(x.size())
#     # cnn = nn.Conv1d()
#     return 0

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
    word_vecs, k = load_bin_vec("GoogleNews-vectors-negative300.bin", vocab)
    print("word2vec loaded!")

    print("num words already in word2vec: " + str(len(word_vecs)))    # 16448

    # Embedding layer
    embedding = nn.Embedding(len(vocab)+1, k, padding_idx=0)
    W = {}                                                      # torch.Size([18765, 300])
    word_idx_map = {}
    W["rand"] = W["vec"] = embedding(torch.LongTensor(range(len(vocab)+1)))
    for word, i in zip(vocab, range(1,len(vocab)+1)):
        if word in word_vecs:
            W["vec"][i] = word_vecs[word]
        word_idx_map[word] = i
    print("dataset created!")

    non_static = [True, False, True]
    U = ["rand", "vec", "vec"]
    results = []
    train_x_idx, train_y, test_x_idx, test_y = make_idx_data(revs, word_idx_map, max_l=max_l, k=300)    # 9595 1067 X 65
    for i in range(3):
        train_x = make_data(train_x_idx, W[U[i]], max_l=max_l, k=300)
        break
    #     perf = train_conv_net(train_x, train_y, W[U[i]], non_static[i], h=[3,4,5], feature=100, p=0.5, s=3, batch=50, k=300)
    #     results.append(perf)
    # with open("results", "w") as f:
    #     f.write(str(results))

main()
