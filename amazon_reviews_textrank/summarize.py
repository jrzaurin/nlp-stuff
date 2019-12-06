import numpy as np
import networkx as nx
import random

from sklearn.metrics.pairwise import cosine_similarity

class Summarizer(object):
	"""
	Simplest pagerank-based summarizer one can possibly think of.

	Parameters:
	----------

	reviewtextDF: pd.DataFrame with the original review and the processed
		sentences
	idx2sent: Dict
		keys are indexes and values sentences
	sent2vec: Dict
		keys are sentence indexes and values are average word vectors

	Attributes:
	----------
	sent2idx: Dict
		keys are sentences and values are indexes
	"""
	def __init__(self, reviewtextDF, idx2sent, sent2vec):
		super(Summarizer, self).__init__()
		self.reviewtextDF = reviewtextDF
		self.idx2sent = idx2sent
		self.sent2idx = {k:v for v,k in idx2sent.items()}
		self.sent2vec = sent2vec

	def summarize(self, idx, n=100, ratio=0.2):
		"""
		Summarize the review corresponding to idx using n random sentences to build
		the graph. The length of the summary will be ratio times the length of the
		review
		"""

		# the indexes of the sentences in the review will be the first ones in the
		# similarity matrix
		org_review = self.reviewtextDF.iloc[idx].reviewText
		proc_sents = self.reviewtextDF.iloc[idx].processed_sents
		sentidx  = [self.sent2idx[s] for s in proc_sents]
		idxmap = {k:v for k,v in enumerate(sentidx)}

        # sample n random sentences to build a graph since 1.4 mil don't fit
        # in memory.
		rand_idx = random.sample(range(1, len(self.sent2idx)), n)
		rand_idx = [r for r in rand_idx if r not in sentidx]
		graph_idx = sentidx + rand_idx

		# compute similarity mtx, corresponding graph and scores
		wvmtx = np.vstack([self.sent2vec[i] for i in graph_idx])
		sim_mat = cosine_similarity(wvmtx) - np.eye(len(wvmtx))
		nx_graph = nx.from_numpy_array(sim_mat)
		scores = nx.pagerank_numpy(nx_graph)

		# extract the indexes corresponding to the sentences in the review and sort them
		scores = list({k: scores[k] for k in idxmap}.items())
		scores = sorted(scores, key=lambda x: x[1], reverse=True)

		# print summary
		summary_len = max(round((len(sentidx) * ratio)), 1)
		top_score_idx = [scores[i][0] for i in range(summary_len)]
		top_real_idx  = [idxmap[i] for i in top_score_idx]
		summary = '\n'.join([self.idx2sent[i] for i in top_real_idx])

		return org_review, summary
