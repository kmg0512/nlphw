import torch
from random import *
from collections import Counter
import argparse
import math


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def cosine(v1, v2):
    return torch.sum(v1 * v2) / torch.sqrt(torch.sum(v1 * v1) * torch.sum(v2 * v2))


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


def ngram(word):
    subwords = []
    for i in range(3, 7):
        l = len(word)
        for j in range(-1, l - i + 1):
            if j == -1:
                subwords.append('<' + word[:i - 1])
            elif j == l - i + 1:
                subwords.append(word[j:] + '>')
            else:
                subwords.append(word[j : j + i])
    subwords.append('<' + word + '>')

    return subwords


def subword_embedding(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index subword of a centerword (type:int)                           #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(V,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    V, D = inputMatrix.size()
    K, _ = outputMatrix.size()

    inputVector = inputMatrix[centerWord]

    grad_out = outputMatrix.clone()

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


def subword_embedding_trainer(input_seq, target_seq, numwords, numsubwords, s2i, stats, NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numsubwords, dimension).cuda() / (dimension**0.5)
    W_out = torch.randn(numwords, dimension).cuda() / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        for inputs, outputs in zip(input_seq,target_seq):
            #Only use the activated rows of the weight matrix
            #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
            activated = torch.tensor([outputs] + [ind for ind in set(sample(stats, NS)) if ind != outputs]).cuda()

            for subword in ngram(inputs):
                i+=1
                L, G_in, G_out = subword_embedding(s2i[subword], W_in, W_out[activated])
                W_in[s2i[subword]] -= learning_rate*G_in.squeeze()
                W_out[activated] -= learning_rate*G_out
                losses.append(L.item())

                if i%2000==0:
                    avg_loss=sum(losses)/len(losses)
                    print("Loss : %f" %(avg_loss,))
                    losses=[]

    return W_in, W_out


def sim(testwords, subword2ind, ind2word, matrix):
    wordsVector = torch.zeros(len(ind2word), matrix.size()[1]).cuda()

    for ind, word in ind2word.items():
        for subword in ngram(word):
            wordsVector[ind] += matrix[subword2ind[subword]]

    for testword in testwords:
        testVector = torch.zeros(1, matrix.size()[1]).cuda()
        for subword in ngram(testword):
            testVector += matrix[subword2ind[subword]]
        distances = [(cosine(testVector, wordsVector[ind]), ind) for ind in len(ind2word)]
        closests = sorted(distances, key=lambda t: t[0], reverse=True)[:5]

        print()
        print("===============================================")
        print("The most similar words to \"" + testword + "\"")
        for dist, ind in closests:
            print(ind2word[ind]+":%.3f"%(dist,))
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
    corpus = subsampling(text.split()) #if part == "full" else text.split()
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    subwords = []
    cnt = 0
    for word in vocab:
        cnt += 1
        subwords += ngram(word)
    subvocab = set(subwords)

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

    s2i = {}
    s2i[" "]=0
    i = 1
    for subword in subvocab:
        s2i[subword] = i
        i+=1
    i2s = {}
    for k,v in s2i.items():
        i2s[v]=k

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
            input_set += [words[j] for _ in range(window_size*2)]
            target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
        elif j>=len(words)-window_size:
            input_set += [words[j] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
        else:
            input_set += [words[j] for _ in range(window_size*2)]
            target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    emb,_ = subword_embedding_trainer(input_set, target_set, len(w2i), len(s2i), s2i, freqtable, NS=ns, dimension=64, epoch=1, learning_rate=0.05)

    testwords = ["narrow-mindedness", "department", "campfires", "knowing", "urbanize", "imperfection", "principality", "abnormal", "secondary", "ungraceful"]
    for tw in testwords:
    	sim(tw,s2i,i2w,emb)

main()