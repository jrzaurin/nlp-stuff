<p align="center">
  <img width="450" src="docs/figures/nlp_stuff_logo.png">
</p>

## NLP stuff

I add here stuff related to NLP.

Within each directory there should be a README file to help guiding you through the code. So far, this is what I have included:

1. `20_newsgroup_classification_cnn_tf`

	This is a dir with very old Tensorflow code. My aim back then was is simply
	to illustrate 3 different ways of building a Convolutional neural network for
	text classification using Tensorflow. Last time I checked (October 2019) The
	code still run, but if you run it you will get every possible warning to
	upgrade. This dir is mostly for me to keep track of the things I do more than
	any other thing.

2. `amazon_reviews_classification_without_DL`

	Predicting the review score for Amazon reviews (Shoes, Clothes and jewelery).
	using tf-idf, LDA and [EnsembleTopics](https://github.com/lmcinnes/enstop)
	along with `lightGBM` and `hyperopt` for the final classification and
	hyper-parameter optimization. I placed special emphasis in the text
	preprocessing.

3. `amazon_reviews_classification_with_EDA`

	Amazon Reviews classification using tf-idf and
	[Easy Data Augmentation](https://github.com/jasonwei20/eda_nlp) along with `lightGBM`
	and `hyperopt` for the final classification and hyper-parameter optimization.
	Following the philosophy of the previous exercise, I placed some emphasis in
	the text preprocessing, in particular in the use of certain tokenizers.

4. `amazon_reviews_textrank`

	The simplest text summarization approach using the `Pagerank` algorithm via
	the	[networkx](https://networkx.github.io/documentation/networkx-1.10/index.html)
	package and comparing the results with the
	[proper](https://github.com/summanlp/textrank) `Textrank` implementation.

5. `rnn_character_tagging`

	Tagging at character level using RNNs with the aim of differentiating for example, different coding languages or writing styles. The code here is based in a [post](http://nadbordrozd.github.io/blog/2017/06/03/python-or-scala/) by [Nadbor](https://www.linkedin.com/in/nadbor-drozd-12316063/)

5. `amazon_reviews_classification_HAN`

	Amazon Reviews classification (score prediction) using Hierarchical Attention Networks ([Zichao Yang, et al., 2016](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)). I have also used a number of Dropout mechanisms from [the work](https://arxiv.org/pdf/1708.02182.pdf) of Stephen Merity, Nitish Shirish Keskar and Richard Socher: Regularizing and Optimizing LSTM Language Models.

Any comments or suggestions please: jrzaurin@gmail.com or even better open an issue.