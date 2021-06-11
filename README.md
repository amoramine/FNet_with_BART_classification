# FNet_with_BART_classification

## Description

FNet proposes to replace Transformers self-attention layers (having `O(n^2)` computational complexity) with linear transformations that mix input tokens. More specifically, the authors find that replacing the attention mechanism with a standard unparameterized Fourier Transform achieves 92% of the accuracy of BERT on the GLUE benchmark, but pre-trains and runs up to seven times faster on GPUs and twice as fast on TPUs. 

This project leverages an existing implementation of FNet's encoder (https://github.com/jaketae/fnet.git) and completes the code to train the model on Stanford Sentiment Treebank (SST) dataset. SST is a dataset aimed for binary classification and is part of GLUE benchmark. 

The original FNet model (from the paper) is pre-trained on masked language modelling (MLM) and next sentence prediction (NSP). However, the paper was only published  one month ago and the pre-trained model is not available online. In order to harness transfer learning and speed up the model's convergence, this project initialises FNet's encoder with BartForSequenceClassification parameters (loaded from HuggingFace). Additionally, BartTokenizer and BartLearnedPositionalEmbedding are used to process the input sequence before it is fed to the encoder. 

In other words, the model can be seen as BART's encoder with a Fourier Transform token mixing mechanism (from FNet) instead of self-attention.

## How to run the project

To run the project, clone the repo and run the following commands: 
1) `cd FNet_with_BART_classification`
2) `pip install -r requirements.txt`
3) `python fnet.py`


## Citation

article{DBLP:journals/corr/abs-2105-03824,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  author    = {James Lee-Thorp and
               Joshua Ainslie and
               Ilya Eckstein and
               Santiago  Ontañón},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  title     = {FNet: Mixing Tokens with Fourier Transforms},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  journal   = {CoRR},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  volume    = {abs/2105.03824},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  year      = {2021},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  url       = {https://arxiv.org/abs/2105.03824},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  archivePrefix = {arXiv},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  eprint    = {2105.03824},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  timestamp = {Fri, 14 May 2021 12:13:30 +0200},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  biburl    = {https://dblp.org/rec/journals/corr/abs-2105-03824.bib},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  bibsource = {dblp computer science bibliography, https://dblp.org}<br/>
}
