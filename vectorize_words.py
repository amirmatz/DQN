import gzip
import json
from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from tqdm import tqdm


def clean_word(word: str):
    word = word.strip()

    if len(word) == 0:
        return ""

    if not word[0].isalpha():
        word = word[1:]

    if len(word) == 0:
        return ""

    if not word[-1].isalpha():
        word = word[:-2]
    return word


def clean_sentence(sentence: str) -> List[str]:
    cleaned_sentence = [clean_word(word) for word in word_tokenize(sentence)]
    return [word for word in cleaned_sentence if word]


class BaseEmbedding(ABC):
    @abstractmethod
    def __getitem__(self, item: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_vocab(self) -> Iterable[str]:
        raise NotImplementedError


class GoogleWord2Vec(BaseEmbedding):
    def __init__(self) -> None:
        with gzip.open("word2vec.json.gz") as f:
            self._d = json.load(f)

        for k, v in self._d.items():
            self._d[k] = np.array(v)

    def __getitem__(self, item: str) -> np.ndarray:
        item = clean_word(item)
        if item in self._d:
            return self._d[item]

        raise NotImplementedError("Bug in design")

    def get_vocab(self) -> Iterable[str]:
        return self._d.keys()


class LightWord2Vec(BaseEmbedding):
    def __init__(self, model: Word2Vec = None) -> None:
        if model is None:
            self._model = LightWord2Vec(Word2Vec.load("train_embedding.model"))  # Default model
        else:
            self._model = model

    @staticmethod
    def train(train_sentences: Iterable[str]):
        sentences = [clean_sentence(sent) for sent in train_sentences]
        print("Calculating word embedding")
        return LightWord2Vec(Word2Vec(tqdm(sentences), min_count=1, workers=8))

    def __getitem__(self, item: str) -> np.ndarray:
        item = clean_word(item)
        if item in self._model.wv:
            return self._model.wv[item]

        raise ValueError(f"No embedding for {item}. Use `encode_sentence`")

    def __contains__(self, item: str) -> bool:
        return item in self._model.wv

    def get_vocab(self) -> Iterable[str]:
        return self._model.wv.vocab.keys()

    def encode_sentence(self, sentence: str) -> List[np.ndarray]:
        cleaned_sentence = clean_sentence(sentence)

        if not all((word in self._model.wv) for word in cleaned_sentence):
            sentence_sequence = [cleaned_sentence]
            self._model.build_vocab(sentence_sequence, update=True)
            self._model.train(sentence_sequence, total_examples=1, epochs=5)

        return [self._model.wv[word] for word in cleaned_sentence]
