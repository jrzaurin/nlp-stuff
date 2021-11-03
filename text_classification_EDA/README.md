## Amazon reviews classification using tfidf and Topic Modeling

Standard text classification techniques using our good old friend *tf-idf* and the [Easy Augmentation Techniques](https://github.com/jasonwei20/eda_nlp) by Jason Wei and Kai Zou.

The order of the `.py` scripts is:

00. `prepare_data.py` : simple manipulation and train/valid/test split
01. `augment.py`: using EDA for data augmentation
02. `feature_extraction.py` : using tf-idf to compute features
04. `score.py`: using Lightgbm with the computed features to predict the score of the review

Easy.

The main addition in this directory relative to the `amazon_reviews_classification_without_DL` dir is the use of Fastai's Tokenizer and EDA.

The results show that while EDA is not really adding much to this datasets (which is expected), you should almost certainly use the Fastai Tokenizer if you can.

There are explanatory notebooks in the `notebooks` dir.

