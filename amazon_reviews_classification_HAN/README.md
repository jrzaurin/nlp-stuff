## Predicting Amazon reviews score using Hierarchical Attention Networks (HAN)

The code in this directory is my implementation, using `Pytorch` and `MxNet`,
of the nice paper *Hierarchical Attention Networks for Document
Classification* ([Zichao Yang et al.,
2016](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf))

I have also used the results and code presented in *Regularizing and
Optimizing LSTM Language Models* ([Stephen Merity, Nitish Shirish Keskar and
Richard Socher., 2017](https://arxiv.org/pdf/1708.02182.pdf)). Namely, their
implementation of Embedding Dropout, Locked Dropout and Weight Dropout. The
later is in itself based on the work of [Wan et al
(2013)](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf):
*Regularization of neural networks using dropconnect*.


### 1. Prepare the data

To run the experiments in this directory you need to prepare the data first.
Assuming you have downloaded the data and place them in the adequate
directory, simply run:

```python
python prepare_data.py
```

### 2. Running the experiments

The main two modules in this directory are: `utils` and `models`.

`utils` contains a series of utilities to prepare the data for the HAN.

`models` contains the `Pytorch` and `Mxnet` implementations of HAN along with some Dropout mechanisms used for the `Pytorch` implementation.

The main files to run the models are: `main_pytorch.py` and `main_mxnet.py`. I
have run 60 experiments that are detailed in `run_experiments.py`. One example
might be:

```python
python main_pytorch.py --batch_size 64 --embed_dim 300 --word_hidden_dim 64 --sent_hidden_dim 64 \
--embed_drop 0.2 --weight_drop 0.2 --locked_drop 0.2 --last_drop 0.2 \
--lr_scheduler reducelronplateau --lr_patience 2 --patience 4 --save_results
```

This will run the `Pytorch` implementation with batch size of 64 reviews, word
embedding dimension of 300, hidden dimension of 64 for both the word and
sentence GRUs, using 0.2 dropout for all, embedding dropout, weight dropout,
locked dropout and last dropout, using a ReduceLROnPlateau learning rate
scheduler with patience 2 and an early stop patience of 4.

For more details on the meaning of these (and other) parameters please, read
the paper and/or have a look to the code in this directory.

### 3. Notebooks

In addition to all of the above, I have included 5 notebooks that are meant to
guide you through the implementation in detail. These are:

1. `Data_Preparation.ipynb`

2. `HAN_implementation.ipynb`

3. `Running_the_HAN.ipynb`

4. `Review_Score_Prediction_Results.ipynb`

5. `Visualizing_Attention.ipynb`

**NOTE**: the notebooks focus on the `Pytorch` implementation. I believe that,
from them, understanding the `Mxnet` implementation should be pretty
straightforward.

Any comments or suggestions please: jrzaurin@gmail.com or even better open an issue.