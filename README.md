```
Aim: 
Use the transformer encoder architecture and build and train a model from scratch with dailydialog dataset

Main Idea:
send the data through the encoder, then classify it using another linear layer and train, validate and test the model with the help of emotion labels. 

Process:
1) Transformer part - Have used the usual transformer architecture from attention is all you need paper, which is - positional encoding, normalization and addition of heads' output and original tensors, feeding forward and again addition and normalization.
4 transformer blocks, 4 heads, 128 size embedding and other parameters as used in the code

2) preprocesssing part - 
• creating a source vocabulary and token ids from the whole data set
• The main challenge was that the data was not clean and is a bunch of sentences in each cell, so needed to break each dialog into sentences and find their corresponding label and feed them into the encoder. 
• loading and batching the data
• padding and adding special tokens in the front and the end and create a mask and send it through the encoder and classify and train it. 
```
