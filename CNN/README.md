# CNN
## CNN for Sentence Classification

|                      | Model            |  MR      |
|----------------------|------------------|----------|
| Random Initialized   | CNN-rand         | 76.1     |
| Word2vec fixed       | CNN-static       | 81.0     |
| Word2vec fine-tuned  | CNN-non-static   | **81.5** |
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
