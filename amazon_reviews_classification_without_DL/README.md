## Amazon reviews classification using tfidf and Topic Modeling

Standard text classification techniques using our good old friend *tf-idf* and topic modeling, namely LDA and the [EnsTop](https://github.com/lmcinnes/enstop) package.

The order of the `.py` scripts is:

00. `prepare_data.py` : simple manipulation
01. `preprocessing.py`: using a number of tokenization techniques
02. `train_test_split.py` : you know...
03. `feature_extraction.py` : using tf-idf, lda and enstop to compute features
04. `score.py`: using Lightgbm with the computed features to predict the score of the review

Easy.

There are explanatory notebooks in the `notebooks` dir
