import torch
from random import *
from collections import Counter
import argparse
import time
import math
import csv
import re


def softmax(o):
    m = torch.max(o)
    e = torch.exp(o - m)
    softmax = e / torch.sum(e)

    return softmax


def text_classification(contextWords, label, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                         #
# label : News label of contextwords (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Weight matrix of output (type:torch.tesnor(K,D))                 #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(V,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    K, _ = outputMatrix.size()

    inputVector = inputMatrix[contextWords].sum(0)
    output = outputMatrix.mm(inputVector.reshape(D, 1))
    y = softmax(output)

    loss = -torch.log(y[label])

    e = y
    e[label] -= 1

    grad_in = torch.mm(e.t(), outputMatrix)
    grad_out = torch.mm(e, inputVector.reshape(1, D))

    return loss, grad_in, grad_out


def text_classification_trainer(input_seq, target_seq, numwords, numlabels, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numlabels, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()

    times = []

    for _ in range(epoch):
        start_time = time.time()

        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1

            L, G_in, G_out = text_classification(inputs, output, W_in, W_out)
            W_in[inputs] -= learning_rate*G_in
            W_out -= learning_rate*G_out

            losses.append(L.item())
            if i%30000==0:
                avg_loss=sum(losses)/len(losses)
                elapsed_time = time.time() - start_time
                print("Loss : %f, Time : %f sec" %(avg_loss, elapsed_time,))
                losses=[]
                start_time = time.time()
                times.append(elapsed_time)

    print()
    print("Total Time : ", sum(times), " sec")
    print("Average Time : ", sum(times) / len(times), " sec")
    print()

    return W_in, W_out


def classify(inputMatrix, outputMatrix, w2i, classes):
    V, D = inputMatrix.size()
    K, _ = outputMatrix.size()

    print("TEST")
    rawcsv = None
    with open('./ag_news_csv/test.csv', 'r') as f:
        rawcsv = list(csv.reader(f))

    f = open("result.txt", 'w')
    f.write("Prediction\tAnswer\n")

    cnt = 0
    ans = 0
    table = torch.zeros(K, K)
    for article in rawcsv:
        article[0] = int(article[0]) - 1
        article[1] = re.sub("[\W_]", ' ', article[1]).split()
        article[2] = re.sub("[\W_]", ' ', article[2]).split()


        input_set = list(set([w2i.get(word) for word in article[1] + article[2] if w2i.get(word) != None]))

        inputVector = inputMatrix[input_set].sum(0)
        output = outputMatrix.mm(inputVector.reshape(D, 1))
        y = softmax(output)

        label = torch.argmax(y)

        fstr = "%s\t%s\n" % (classes[label], classes[article[0]])
        f.write(fstr)

        if label == article[0]:
            ans += 1
        cnt += 1

        table[label][article[0]] += 1

    f.write("\nRow: Prediction, Column: Answer\n")
    for t in table:
        f.write(str(t) + '\n')

    fstr = "\nResult: %d / %d (%0.2f%%)\n" % (ans, cnt, (ans / cnt * 100))
    f.write(fstr)
    f.close()

    print("\nRow: Prediction, Column: Answer\n")
    print(table)
    print(fstr)



def main():
	#Load and preprocess corpus
    print("loading...")
    rawcsv = None
    with open('./ag_news_csv/train.csv', 'r') as f:
        rawcsv = list(csv.reader(f))

    classes = None
    with open('./ag_news_csv/classes.txt', 'r') as f:
        classes = [word.strip() for word in f.readlines()]

    print("preprocessing...")
    articles = []
    corpus = []
    for article in rawcsv:
        article[0] = int(article[0]) - 1
        article[1] = re.sub("[\W_]", ' ', article[1]).split()
        article[2] = re.sub("[\W_]", ' ', article[2]).split()
        corpus += article[1] + article[2]
        articles.append(article)
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    #Make training set
    print("build training set...")
    input_set = []
    target_set = []
    for article in articles:
        input_set.append(list(set([w2i.get(word) for word in article[1] + article[2] if w2i.get(word) != None])))
        target_set.append(article[0])

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    im, om = text_classification_trainer(input_set, target_set, len(w2i), len(classes), dimension=64, epoch=1, learning_rate=0.05)
    classify(im, om, w2i, classes)

main()