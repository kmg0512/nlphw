import torch
from random import *
from collections import Counter
import argparse
import time
import math

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def subsampling(word_seq):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################

    subsampled=[]
    t = 10e-5
    f = Counter(word_seq)
    l = len(word_seq)

    for w in word_seq:
        p = 1 - math.sqrt(t / (f[w] / l))
        if p <= random():
            subsampled.append(w)

    return subsampled


def skipgram_NS(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    K, _ = outputMatrix.size()

    loss = None
    grad_in = None
    grad_out = torch.Tensor(K, D)

    inputVector = inputMatrix[centerWord]

    p = sigmoid(torch.mm(outputMatrix[0].reshape(1, D), inputVector.reshape(D, 1)))
    loss = -torch.log(p)
    grad_in = -(1 - p) * outputMatrix[0]
    grad_out[0] = -(1 - p) * inputVector

    for k in range(1, K):
        q = sigmoid(-torch.mm(outputMatrix[k].reshape(1, D), inputVector.reshape(D, 1)))
        loss -= torch.log(q)
        grad_in += (1 - q) * outputMatrix[k]
        grad_out[k] = (1 - q) * inputVector

    return loss, grad_in, grad_out


def subword_embedding_trainer(input_seq, target_seq, numwords, stats, NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
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
            #Only use the activated rows of the weight matrix
            #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
            activated = [output] + sample(stats, NS)
            L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
            W_in[inputs] -= learning_rate*G_in.squeeze()
            W_out[activated] -= learning_rate*G_out

            losses.append(L.item())
            if i%50000==0:
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


def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)

    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()


def main():
    parser = argparse.ArgumentParser(description='Subword Embedding')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    part = args.part
    ns = args.ns

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = subsampling(text.split()) if part == "full" else text.split()
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    subvocab = []
    for word in vocab:
        for i in range(3, 7):
            l = len(word)
            for j in range(-1, l - i + 1):
                if j == -1:
                    subvocab.append('<' + word[:i - 1])
                elif j == l - i + 1:
                    subvocab.append(word[j:] + '>')
                else:
                    subvocab.append(word[j : j + i])
        subvocab.append('<' + word + '>')

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for subword in subvocab:
        w2i[subword] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    #Make training set
    print("build training set...")
    input_set = []
    target_set = []
    window_size = 5
    for j in range(len(words)):
        if j<window_size:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
        elif j>=len(words)-window_size:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
        else:
            input_set += [w2i[words[j]] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print(vocab)

    #Training section
    emb,_ = subword_embedding_trainer(input_set, target_set, len(w2i), freqtable, NS=ns, dimension=64, epoch=1, learning_rate=0.025)

    testwords = ["narrow-mindedness", "department", "campfires", "knowing", "urbanize", "imperfection", "principality", "abnormal", "secondary", "ungraceful"]
    for tw in testwords:
    	sim(tw,w2i,i2w,emb)

main()