# CNN


### For Sentence Classification
- Nonlinearity: ReLU
- Window filter sizes h = 3, 4, 5
- Each filter size has 100 feature maps
- Dropout p = 0.5
- L2 constraint s for rows of softmax, s = 3
- Mini batch size for SGD training: 50
- Word vectors: pre-trained with word2vec, k = 300
- During training, keep checking performance on dev set and pick highest accuracy weights for final evaluation


### Implementation
#### MR dataset
- Movie reviews with sentiment polarity (positive / negative)
- Pre-processed
#### You need to build vocabulary to give an index number to a word
| Word    | Index |
|---------|-------|
| rock    | 0     |
| century | 1     |
| new     | 2     |
| and     | 3     |
| ...     | ...   |
#### Embedding layer
```python
CLASS torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
```
A simple lookup table that stores embeddings of a fixed dictionary and size.
This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.

Parameters
- num_embeddings (int) – size of the dictionary of embeddings
- embedding_dim (int) – the size of each embedding vector
- padding_idx (int, optional) – If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.
- max_norm (float, optional) – If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
- norm_type (float, optional) – The p of the p-norm to compute for the max_norm option. Default 2.
- scale_grad_by_freq (boolean, optional) – If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.
- sparse (bool, optional) – If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.

Example
```python
# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input)
```
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],
        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])
#### Convolutional Layer
```python
CLASS torch.nn.Conv1d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
```
Parameters
- in_channels(int) = Number of channels in the input image
- out_channels(int) - Number of channels produced by the convolution
- kernel_size(int or tuple) - Size of the convolving kernel
- stride(int or tuple,optional) - Stride of the convolution
- padding(int or tuple,optional) - Zero-padding added to both sides of the input
- padding_mode(string,optional) - zeros
- dilation(int or tuple,optional) - Spacing between kernel elements
- groups(int,optional) - Number of blocked connections from input channels to output channels
- bias(bool,optional) - If True, adds a learnable bias to the output
#### Pooling Layer
```python
CLASS torch.nn.AdaptiveMaxPool1d(output_size, return_indices=False)
```
Applies a 1D adaptive max pooling over an input signal composed of several input planes.
The output size is H, for any input size. The number of output features is equal to the number of input planes.

Parameters
- output_size - the target output size H
- return_indices - if True, will return  the indices along with the outputs. Useful to pass to nn.MaxUnpool1d.
#### FC Layer
```python
CLASS torch.nn.Linear(in_features, out_features, bias=True)
```
Applies a linear transformation to the incoming data: y=xA^T+b
Parameters
- in_features - size of each input sample
- out_features - size of each output sample
- bias - If set to False, the layer will not learn an additive bias
Example
```python
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
```
torch.Size([128, 30])
#### Other layers
##### Dropout
```python
CLASS torch.nn.Dropout(p=0.5, inplace=False)
```
##### ReLU
```python
CLASS torch.nn.ReLu(inplace=False)
```
Applies the recrified linear unit function element-wise
- `ReLU(x) = max(0,x)`

Parameters
- inplace - can optionally do the operation in-place
#### Framework
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py

## CNN for Sentence Classification
|                      | Model            |  MR      |
|----------------------|------------------|----------|
| Random Initialized   | CNN-rand         | 76.1     |
| Word2vec fixed       | CNN-static       | 81.0     |
| Word2vec fine-tuned  | CNN-non-static   | __81.5__ |
| Fixed and fine-tuned | CNN-multichannel | 81.1     |
Set requires_grad to false you want to freeze:
```python
# we want to freeze the fc2 Layer
net.fc2.weight.requires_grad = False
net.fc2.bias.requires_grad = False
```
Then set the optimizer like the following:
```python
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
```
Pre-trained word2vec : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

Due: 6. 5.
