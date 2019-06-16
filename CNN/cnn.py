import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm



# Use CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class CNN(nn.Module):
    def __init__(self, word_vecs, model_type, h, feature, k, p, max_l):
        super(CNN, self).__init__()
        v = word_vecs.size()[0]

        # Embedding Layer
        self.ch = 1
        self.emb = nn.Embedding(v, k, padding_idx=0)
        if model_type != "rand":
            self.emb.weight.data.copy_(word_vecs)
            if model_type == "static":
                self.emb.weight.requires_grad = False
            elif model_type == "multichannel":
                self.emb_multi = nn.Embedding(v, k, padding_idx=0)
                self.emb_multi.weight.data.copy_(word_vecs)
                self.emb_multi.weight.requires_grad = False
                self.ch = 2

        # Convolutional Layer
        for w in h:
            conv = nn.Conv1d(self.ch, feature, w * k, stride=k)
            setattr(self, 'conv%d' % w, conv)

        # Pooling Layer
        self.pool = nn.AdaptiveMaxPool1d(1)

        # FC Layer
        self.fc = nn.Linear(len(h) * feature, 2)

        # Other Layers
        self.dropout = nn.Dropout(p)
        self.relu = nn.ReLU()

        self.h = h
        self.feature = feature
        self.k = k
        self.max_l = max_l


    def forward(self, input_x):
        x = self.emb(input_x).reshape(-1, 1, self.max_l * self.k)
        if self.ch == 2:
            x_multi = self.emb_multi(input_x).reshape(-1, 1, self.max_l * self.k)
            x = torch.cat((x, x_multi), dim=1)

        outs = []
        for w in self.h:
            conv = getattr(self, 'conv%d' % w)
            out = self.dropout(self.relu(conv(x)))
            out = self.pool(out)
            outs.append(out)
        outs = torch.cat(outs, dim=1).reshape(-1, len(self.h) * self.feature)
        output = self.fc(outs)
        return output



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
    fnames = ["rt-polarity.neg", "rt-polarity.pos"]
    data = []
    vocab = defaultdict(int)
    max_l = 0

    for i in range(2):
        with open(fnames[i], "r", encoding="cp1252") as f:
            for line in f:
                sent = clean_str(line.strip())
                
                words = set(sent.split())
                for word in words:
                    vocab[word] += 1

                datum  = {"y":i, "text": sent}
                data.append(datum)
                
                l = len(sent.split())
                if l > max_l:
                    max_l = l

    return data, vocab, max_l


def load_bin_vec(fname, vocab):
    pre_trained = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, k = map(int, header.split())  # 3000000, 300
        binary_len = 4 * k

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
                pre_trained[word] = torch.from_numpy(np.frombuffer(f.read(binary_len), dtype='float32')).to(device)
            else:
                f.read(binary_len)

    return pre_trained, k


def get_idx_from_sent(sent, word_idx_map, max_l=56, k=300):
    word_idx = []
    
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            word_idx.append(word_idx_map[word])

    # Padding
    while len(word_idx) < max_l:
        word_idx.append(0)

    return word_idx

def cnn_trainer(train_loader, test, word_vecs, max_l, model_type, h=[3,4,5], feature=100, p=0.5, s=3, k=300, epoch=1000, learning_rate=0.1):
    model = CNN(word_vecs, model_type, h, feature, k, p, max_l).to(device)
    model.fc.weight.requires_grad = False
    model.fc.bias.requires_grad = False

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Train model
    for i in tqdm(range(epoch), desc='epoch', leave=False):
        losses = []
        for train_x, train_y in tqdm(train_loader, desc='train', leave=False):
            train_x, train_y = Variable(train_x), Variable(train_y)
            optimizer.zero_grad()
            output = model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=s)
            optimizer.step()
            losses.append(loss.item())
        if (i+1) % 5 == 0:
            print("Epoch: ", i+1, "Loss: ", )

    # Test model
    test_x, test_y = test.tensors
    test_x, test_y = Variable(test_x), Variable(test_y)
    result = torch.argmax(model(test_x), dim=1)
    accuracy = torch.mean((test_y == result).float())

    return accuracy

def main():
    # Read MR dataset
    print("loading data...", end=' ')
    data, vocab, max_l = build_data()
    print("data loaded!")

    print("number of sentences: " + str(len(data))) # 10662
    print("vocab size: " + str(len(vocab)))         # 18764
    print("max sentence length: " + str(max_l))     # 56

    # Read pre-trained word2vec
    print("loading word2vec vectors...", end=' ')
    pre_trained, k = load_bin_vec("GoogleNews-vectors-negative300.bin", vocab)
    print("word2vec loaded!")

    print("num words already in word2vec: " + str(len(pre_trained)))    # 16448

    # Make word vectors
    word_vecs = [torch.zeros(1, k).to(device)] # torch.Size([18765, 300])
    word_idx_map = {}
    for i, word in enumerate(vocab):
        if word in pre_trained:
            word_vecs.append(pre_trained[word].reshape(1, k))
        else:
            word_vecs.append(torch.randn(1, k).to(device))
        word_idx_map[word] = i + 1
    word_vecs = torch.cat(word_vecs)
    print("dataset created!")

    # Initialize parameters
    model_type = ["rand", "static", "non-static", "multichannel"]
    h = [3, 4, 5]
    feature = 100
    p = 0.5
    s = 3

    # Make Train/Test dataset
    x, y = [], []
    random.shuffle(data)
    for datum in data:
        word_idx = get_idx_from_sent(datum["text"], word_idx_map, max_l, k)
        x.append(word_idx)
        y.append(datum["y"])

    split_idx = int(len(data) * 0.9) # 9595 1067 X 56
    train_x, train_y = torch.LongTensor(x[:split_idx]).to(device), torch.LongTensor(y[:split_idx]).to(device)
    test_x, test_y = torch.LongTensor(x[split_idx:]).to(device), torch.LongTensor(y[split_idx:]).to(device)

    train, test = TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)
    train_loader = DataLoader(train, batch_size=50)

    # Train
    for i in tqdm(range(4), desc='model_type', leave=False):
        acc = cnn_trainer(train_loader, test, word_vecs, max_l, model_type[i], h=h, feature=feature, p=p, s=s, k=k, epoch=25, learning_rate=0.1)
        print(model_type[i], acc * 100, '%')

main()
