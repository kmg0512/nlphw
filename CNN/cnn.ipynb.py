#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'CNN'))
	print(os.getcwd())
except:
	pass

#%%
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


#%%
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

#%%
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

#%%
# read MR dataset
print("loading data...", end=' ')
revs, vocab, max_l = build_data()
print("data loaded!")

print("number of sentences: " + str(len(revs)))             # 10662
print("vocab size: " + str(len(vocab)))                     # 18764
print("max sentence length: " + str(max_l))                 # 56

#%%
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())  # 3000000, 300
        binary_len = 4 * layer1_size
        for line in tqdm(range(vocab_size), desc='load_bin_vec'):
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

#%%
# read pre-trained word2vec
print("loading word2vec vectors...", end=' ')
word_vecs, k = load_bin_vec("GoogleNews-vectors-negative300.bin", vocab)
print("word2vec loaded!")

print("num words already in word2vec: " + str(len(word_vecs)))    # 16448

#%%
# Embedding layer
embedding = nn.Embedding(len(vocab)+1, k, padding_idx=0)
W = {}
word_idx_map = {}
W["rand"] = W["vec"] = embedding(torch.LongTensor(range(len(vocab)+1))) # torch.Size([18765, 300])
for word, i in zip(vocab, range(1,len(vocab)+1)):
    if word in word_vecs:
        W["vec"][i] = word_vecs[word]
    word_idx_map[word] = i
print("dataset created!")

#%%
def get_idx_from_sent(sent, word_idx_map, max_l=56, k=300):
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

#%%
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
    train_y = torch.LongTensor(train_y)
    test_y = torch.LongTensor(test_y)
    return train_x_idx, train_y, test_x_idx, test_y

#%%
def make_data(x_idx, W, max_l=56, k=300):
    x = torch.Tensor(len(x_idx), 1, max_l * k)
    for i, sent in enumerate(x_idx):
        xx = []
        for idx in sent:
            xx.append(W[idx])
        x[i] = torch.cat(tuple(xx))
    return x

#%%
class CNN(nn.Module):
    def __init__(self, hs, feature, k, p):
        super(CNN, self).__init__()
        for h in hs:
            conv = nn.Conv1d(1, feature, h * k, stride=k)
            setattr(self, 'conv%d' % h, conv)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(len(hs) * feature, 2)
        self.loss = nn.LogSoftmax(dim=-1)
        self.hs = hs

    def forward(self, x):
        outs = []
        for h in self.hs:
            conv = getattr(self, 'conv%d' % h)
            out = self.drop(self.relu(conv(x)))
            out = self.pool(out)
            outs.append(out)
        outs = torch.cat(outs, dim=1).reshape(-1, 300)
        outs = self.fc(outs)
        return self.loss(outs)

#%%
def cnn_trainer(train_loader, test_x, test_y, W, non_static, h=[3,4,5], feature=100, p=0.5, s=3, k=300):
    criterion = nn.CrossEntropyLoss()
    model = CNN(h, feature, k, p)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    total_loss=0
    for epoch in tqdm(range(10), desc='epoch', leave=False):
        total_loss = 0
        for train_x, train_y in tqdm(train_loader, desc='train', leave=False):
            train_x, train_y = Variable(train_x), Variable(train_y)
            optimizer.zero_grad()
            output = model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
    print(total_loss)

    test_x, test_y = Variable(test_x), Variable(test_y)
    result = torch.max(model(test_x).data, 1)[1]
    accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())

    return accuracy

#%%
non_static = [True, False, True]
U = ["rand", "vec", "vec"]
accuracies = []
train_x_idx, train_y, test_x_idx, test_y = make_idx_data(revs, word_idx_map, max_l=max_l, k=k)    # 9595 1067 X 56
for i in tqdm(range(3), desc='i', leave=False):
    train_x = make_data(train_x_idx, W[U[i]], max_l=max_l, k=k) # 9595, 1, 16800
    test_x = make_data(test_x_idx, W[U[i]], max_l=max_l, k=k)
    train = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train, batch_size=50)
    accuracy = cnn_trainer(train_loader, test_x, test_y, W[U[i]], non_static[i], h=[3,4,5], feature=100, p=0.5, s=3, k=k)
    accuracies.append(accuracy)
    print(accuracy)

#%%
print(accuracies)