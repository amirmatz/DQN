from typing import Iterable

import config

SOS_TOKEN_INDEX = 0
EOS_TOKEN_INDEX = 1

ACTIONS = list({
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
})


class Lang:
    def __init__(self, vocab: Iterable[str]):
        self.word2index = {}
        self.index2word = {0: config.SOS_TOKEN, 1: config.EOS_TOKEN, 2: config.UNKNOWN_TOKEN}
        self.n_words = len(self.index2word)  # Count SOS and EOS and UNK

        for word in vocab:
            self.add_word(word)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def get_actions(self):
        return ACTIONS

    def words(self):
        return (word for word in self.word2index.keys())

    def size(self):
        return self.n_words
