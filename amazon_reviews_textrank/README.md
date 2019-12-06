## Amazon reviews classification using tfidf and Topic Modeling

Here is a relatively quick attempt to build `TextRank`using
[`networkx`](https://networkx.github.io/documentation/networkx-1.10/index.html)
Pagerank. I compare the results with the proper implementation
[here](https://github.com/summanlp/textrank)

As with most of the code throughout this repo, the code is not meant to be
production-ready, but readable so one can see what is happening. You might
find some of the helper function useful for your tasks

The order of the `.py` scripts is:

01. `prepare_data.py` : simple manipulation and sentence tokenization
02. `sentence_vectors.py`: build sentence vectors averaging word vectors
03. `reviews_summary.py` : summarize reviews using the class `Summarizer` at `summarize.py`

Easy.

As one might expect, the `SummaNLP` implementation works better than mine.
There are explanatory notebooks in the `notebooks` dir.

