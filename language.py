import config


SOS_TOKEN_INDEX = 0
EOS_TOKEN_INDEX = 1


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {0: config.SOS_TOKEN, 1: config.EOS_TOKEN}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def get_actions(self):
        pass # todo add special toekns that can be viewed as actions

    def words(self):
        return (word for word in self.word2index.keys())

    def size(self):
        return self.n_words
