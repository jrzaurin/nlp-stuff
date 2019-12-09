## NLP stuff

Here I intend to add stuff related to NLP.

Within each directory there should be a README file to help guiding you through the code. So far, this is what I have included:

1. `20_newsgroup_classification_cnn_tf`

	This is a dir with very old Tensorflow code. My aim was is simply to
	illustrate 3 different ways of building a Convolutional neural network for
	text classification using Tensorflow. The code still runs, but if you run it
	you will get every possible warning to upgrade.

2. `amazon_reviews_classification_without_DL`

	Predicting the review score for Amazon reviews (Shoes, Clothes and jewelery).
	using tf-idf, LDA and [EnsembleTopics](https://github.com/lmcinnes/enstop)
	along with `lightGBM` and `hyperopt` for the final classification and
	hyper-parameter optimization. I placed special emphasis in the text
	preprocessing.

3. `amazon_reviews_classification_with_EDA`

	Amazon Reviews classification using tf-idf and `[Easy Data
	Augmentation]`(https://github.com/jasonwei20/eda_nlp) along with `lightGBM`
	and `hyperopt` for the final classification and hyper-parameter optimization.
	Following the philosophy of the previous exercise, I placed some emphasis in
	the text preprocessing, in particular in the use of certain tokenizers.

4. `amazon_reviews_textrank`

	The simplest text summarization approach using the `Pagerank` algorithm via
	the	[networkx](https://networkx.github.io/documentation/networkx-1.10/index.html)
	package and comparing the results with the
	[proper](https://github.com/summanlp/textrank) `Textrank` implementation.

5. `rnn_character_tagging`

	Tagging at character level using RNNs with the aim of differentiating for example, different coding languages or writing styles.


