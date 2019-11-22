import spacy
import random
import re
import pdb

from random import shuffle
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import wordnet
from tqdm import tqdm

random.seed(1)

tok = spacy.blank('en', disable=["parser","tagger","ner"])


def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in STOP_WORDS]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = _get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1
		if num_replaced >= n: break
	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')
	return new_words


def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words


def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = _swap_word(new_words)
	return new_words


def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		_add_word(new_words)
	return new_words


def _get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word):
		for l in syn.lemmas():
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym)
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)


def _swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
	return new_words


def _add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = _get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)


def rm_useless_spaces(t):
    return re.sub(' {2,}', ' ', t)


class EDA(object):
	def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=3,
		sr=True, ri=True, rs=True, rd=True):
		super(EDA, self).__init__()

		self.alpha_sr = alpha_sr
		self.alpha_ri = alpha_ri
		self.alpha_rs = alpha_rs
		self.p_rd = p_rd
		self.num_aug = num_aug
		self.sr = sr
		self.ri = ri
		self.rs = rs
		self.rd = rd

	def _augment(self, sentence):

		words = [t.text.lower() for t in tok.tokenizer(sentence)]
		words = [word for word in words if word is not '']
		num_words = len(words)

		n_sr = max(1, int(self.alpha_sr*num_words))
		n_ri = max(1, int(self.alpha_ri*num_words))
		n_rs = max(1, int(self.alpha_rs*num_words))

		a_sents = []
		if self.sr: a_sents+= [' '.join(synonym_replacement(words, n_sr)) for _ in range(self.num_aug)]
		if self.ri: a_sents+= [' '.join(random_insertion(words, n_ri)) for _ in range(self.num_aug)]
		if self.rs: a_sents+= [' '.join(random_swap(words, n_rs)) for _ in range(self.num_aug)]
		if self.rd: a_sents+= [' '.join(random_deletion(words, self.p_rd)) for _ in range(self.num_aug)]

		a_sents += [sentence.lower()]
		# just in case as I am going to use this as a separator
		a_sents =  [s.replace(":", "") for s in a_sents]
		a_sents =  [rm_useless_spaces(s) for s in a_sents]
		shuffle(a_sents)

		return a_sents

	def augment(self, train, out_file=None):

		if out_file is not None: writer = open(out_file, 'w')
		else: a_train = []

		for sentence, label in tqdm(train):
			a_sents = self._augment(sentence)
			if out_file is not None:
				for s in a_sents: writer.write(str(label) + "::" + s + '\n')
			else:
				a_train.append( list( zip(a_sents, [label]*len(a_sents)) ) )

		if out_file is not None: writer.close()
		else: return a_train