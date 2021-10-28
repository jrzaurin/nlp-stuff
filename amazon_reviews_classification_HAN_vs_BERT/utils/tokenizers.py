import numpy as np
import spacy
import os


from pytorch_widedeep.utils import Vocab
from sklearn.exceptions import NotFittedError
from transformers import BertTokenizer
from .preprocessing_utils import *  # noqa: F403

n_cpus = os.cpu_count()


class BaseTokenizer(object):
    def __init__(self):
        super(BaseTokenizer, self).__init__()

    def fit(self, texts):
        raise NotImplementedError

    def transform(self, texts):
        raise NotImplementedError

    def fit_transform(self, texts):
        raise NotImplementedError


class BertFamilyTokenizer(object):
    """docstring for BertTokenizer"""

    def __init__(
        self,
        pretrained_tokenizer="bert-base-uncased",
        do_lower_case=True,
        max_length=128,
    ):
        super(BertFamilyTokenizer, self).__init__()
        self.pretrained_tokenizer = pretrained_tokenizer
        self.do_lower_case = do_lower_case
        self.max_length = max_length

    def fit(self, texts):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_tokenizer, do_lower_case=self.do_lower_case
        )

    @staticmethod
    def _pre_rules(text):
        return fix_html(rm_useless_spaces(spec_add_spaces(text)))

    def transform(self, texts):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded_sent = self.tokenizer.encode_plus(
                text=self._pre_rules(text),
                add_special_tokens=True,
                max_length=self.max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
            )

            input_ids.append(encoded_sent.get("input_ids"))
            attention_masks.append(encoded_sent.get("attention_mask"))

        return np.stack(input_ids), np.stack(attention_masks)

    def fit_transform(self, texts):
        self.fit(texts).transform(texts)


class HANTokenizer(BaseTokenizer):
    def __init__(
        self,
        max_vocab=30000,
        batch_size=1000,
        min_freq=5,
        q=0.8,
        with_preprocess=False,
        pad_sent_first=True,
        pad_doc_first=False,
        pad_idx=1,
        n_cpus=None,
        verbose=True,
    ):
        super(HANTokenizer, self).__init__()
        self.max_vocab = max_vocab
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.q = q
        self.with_preprocess = with_preprocess
        self.pad_sent_first = pad_sent_first
        self.pad_doc_first = pad_doc_first
        self.pad_idx = pad_idx
        self.n_cpus = os.cpu_count() if n_cpus is None else n_cpus
        self.verbose = verbose

        self.tok_func = spacy.load("en_core_web_sm")

    def _sentencizer(self, texts):
        texts_sents = []
        for doc in self.tok_func.pipe(
            texts, n_process=self.n_cpus, batch_size=self.batch_size
        ):
            sents = [str(s) for s in list(doc.sents)]
            texts_sents.append(sents)
        return texts_sents

    def fit(self, texts):
        if self.verbose:
            print("Running sentence tokenizer for {} documents...".format(len(texts)))
        texts_sents = self._sentencizer(texts)
        # from nested to flat list. For speed purposes
        all_sents = [s for sents in texts_sents for s in sents]
        # saving the lengths of the documents: 1) for padding purposes and 2) to
        # compute consecutive ranges so we can "fold" the list again
        texts_length = [0] + [len(s) for s in texts_sents]
        range_idx = [sum(texts_length[: i + 1]) for i in range(len(texts_length))]
        if self.verbose:
            print("Tokenizing {} sentences...".format(len(all_sents)))
        sents_tokens = get_texts(all_sents, self.with_preprocess)
        # saving the lengths of sentences for padding purposes
        sents_length = [len(s) for s in sents_tokens]
        try:
            self.vocab
            if self.verbose:
                print("Using existing vocabulary")
        except AttributeError:
            if self.verbose:
                print("Building Vocab...")
            self.vocab = Vocab.create(
                sents_tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
            )
        # 'numericalize' each sentence
        sents_numz = [self.vocab.numericalize(s) for s in sents_tokens]
        # group the sentences again into documents
        texts_numz = [
            sents_numz[range_idx[i] : range_idx[i + 1]]
            for i in range(len(range_idx[:-1]))
        ]
        # compute max lengths for padding purposes
        self.maxlen_sent = int(np.quantile(sents_length, q=self.q))
        self.maxlen_doc = int(np.quantile(texts_length[1:], q=self.q))

        if self.verbose:
            print("Padding sentences and documents...")
        padded_texts = [
            pad_nested_sequences(
                r,
                self.maxlen_sent,
                self.maxlen_doc,
                pad_sent_first=self.pad_sent_first,
                pad_doc_first=self.pad_doc_first,
                pad_idx=self.pad_idx,
            )
            for r in texts_numz
        ]
        return np.stack(padded_texts, axis=0)

    def transform(self, texts):
        try:
            self.vocab
        except AttributeError:
            raise NotFittedError(
                "This HANTokenizer instance is not trained yet. "
                "Call 'tokenize' with appropriate arguments before using this estimator."
            )
        return self.tokenize(texts)

    def fit_transform(self, texts):
        self.fit(texts).transform(texts)
