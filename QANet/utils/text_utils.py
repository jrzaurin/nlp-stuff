import numpy as np
import pickle

from collections import defaultdict, Counter


PAD, UNK = "xxpad", "xxunk"
text_spec_tok = [PAD, UNK]


class Vocab:
    "Contain the correspondence between numbers and tokens and numericalize."

    def __init__(self, itos):
        self.itos = itos
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(self.itos)})

    def numericalize(self, t):
        "Convert a list of tokens t to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums, sep=" "):
        "Convert a list of nums to their tokens."
        return (
            sep.join([self.itos[i] for i in nums])
            if sep is not None
            else [self.itos[i] for i in nums]
        )

    def __getstate__(self):
        return {"itos": self.itos}

    def __setstate__(self, state):
        self.itos = state["itos"]
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(self.itos)})

    def save(self, path):
        "Save self.itos in path"
        pickle.dump(self.itos, open(path, "wb"))

    @classmethod
    def create(cls, container, max_vocab=50000, min_freq=5):
        "Create a vocabulary from a set of tokens in container"
        if isinstance(container, Counter):
            itos = [o for o, c in container.most_common(max_vocab) if c >= min_freq]
        else:
            itos = list(container)
        for o in reversed(text_spec_tok):
            if o in itos:
                itos.remove(o)
            itos.insert(0, o)
        itos = itos[:max_vocab]
        return cls(itos)

    @classmethod
    def load(cls, path):
        "Load the Vocab contained in path"
        itos = pickle.load(open(path, "rb"))
        return cls(itos)