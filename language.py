from collections import defaultdict
from typing import Iterable

import config

SOS_TOKEN_INDEX = 0
EOS_TOKEN_INDEX = 1
UNK_TOKEN_INDEX = 2

ACTIONS = list(word for sent in {
    "if", "is", "are",
    'number of', 'highest', 'largest', 'lowest', 'smallest', 'maximum', 'minimum',
    'max', 'min', 'sum', 'total', 'average', 'avg', 'mean',
    'for each',
    "where",
    'same as', 'higher than', 'larger than', 'smaller than', 'lower than',
    'more', 'less', 'at least', 'at most', 'equal', 'is', 'are', 'was', 'contain',
    'include', 'has', 'have', 'end with', 'start with', 'ends with',
    'starts with', 'begin',
    'highest', 'largest', 'most', 'smallest', 'lowest', 'smallest', 'least',
    'longest', 'shortest', 'biggest',
    "both", "and",
    "besides",
    'not in', 'sorted by', 'order by',
    'ordered by',
    'which', 'and', ',', 'sum', 'difference', 'multiplication', 'division',
    '#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9', '#10'
} for word in sent.split())


def canonize(word: str) -> str:
    return "".join([c for c in word.lower() if c.isalpha() or c.isdigit()])


class Lang:
    def __init__(self, vocab: Iterable[str]):
        self.word2index = defaultdict(lambda: UNK_TOKEN_INDEX)  # TODO: rethink, a bit extreme
        self.index2word = {0: config.SOS_TOKEN, 1: config.EOS_TOKEN, 2: config.UNK_TOKEN}
        for word in self.get_actions():
            self.add_sentence(word)
        with open("vocab.txt", encoding='utf8') as f:
            for line in f.readlines():
                self.add_sentence(line)
        for word in vocab:
            self.add_sentence(word)

    @property
    def n_words(self):
        return len(self.index2word)

    def add_sentence(self, sentence):
        for sub_sent in sentence.split():
            for word in sub_sent.split("-"):
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word.lower()

    def word_to_index(self, word):
        if word in self.word2index:
            return self.word2index[word]

        canonized_word = canonize(word)
        if canonized_word == "":
            raise ValueError(word)

        return self.word2index[canonized_word]

    def get_actions(self):
        return ACTIONS

    def words(self):
        return (word for word in self.word2index.keys())

    def size(self):
        return self.n_words
