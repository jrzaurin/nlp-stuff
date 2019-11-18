## Amazon reviews classification using tfidf and Topic Modeling

Standard text classification techniques using our good old friend *tf-idf* and topic modeling, namely LDA and the [EnsTop](https://github.com/lmcinnes/enstop) package.

The order of the `.py` scripts is:

00. `prepare_data.py` : simple manipulation
01. `preprocessing.py`: using a number of tokenization techniques
02. `train_test_split.py` : you know...
03. `feature_extraction.py` : using tf-idf, lda and enstop to compute features
04. `score.py`: using Lightgbm with the computed features to predict the score of the review

Easy.

Note that is not the goal of this repo to go into the details of each technique (in case you are interested in a deep dive into LDA have a look [here](https://github.com/jrzaurin/Topic_Modelling_and_Self_Organizing_Maps)). Therefore, I have not used [pyldavis](https://github.com/bmabey/pyLDAvis) to understand the content of the topics, or compare perplexity (and/or coherence) between the LDA and EnsembleTopics (within `EnsTop`). Simply, I just aimed to compare different "standard" techniques used to classify text.

There are explanatory notebooks in the `notebooks` dir.

