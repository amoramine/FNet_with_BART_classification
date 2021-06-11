# FNet_classification

## Description

FNet proposes to replace Transformers self-attention layers (having `O(n^2)` computational complexity) with linear transformations that mix input tokens. More specifically, the authors find that replacing the attention mechanism with a standard unparameterized Fourier Transform achieves 92% of the accuracy of BERT on the GLUE benchmark, but pre-trains and runs up to seven times faster on GPUs and twice as fast on TPUs. 

This project leverages an existing implementation of FNet's encoder (https://github.com/jaketae/fnet.git) and completes the code to train the model on Stanford Sentiment Treebank (SST) dataset. SST is dataset aimed for binary classification and is part of GLUE benchmark. 

The original FNet model (from the paper) is pre-trained on masked language modelling (MLM) and next sentence prediction (NSP). However, the paper was only published  one month ago and the pre-trained model is not available online. In order to harness transfer learning and speed up the Fnet's convergence, the encoder's parameters are initialised using BartForSequenceClassification state_dict from HuggingFace. Additionally, BartTokenizer and BartLearnedPositionalEmbedding are used to process the input sequence before it is fed to the encoder. 

Overall, the model can be seen as BART's encoder with a Fourier Transform token mixing mechanism instead of self-attention.

## How to run the project

To run the project, clone the repo and run the following commands: 
1) `cd FNet_classification`
2) `pip install -r requirements.txt`
3) `python fnet.py`
