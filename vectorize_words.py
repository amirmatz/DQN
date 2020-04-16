import gzip
import json
from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np
import torch
import fasttext

from nltk import word_tokenize


class BaseEmbedding(ABC):
    @abstractmethod
    def __getitem__(self, item: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_vocab(self) -> Iterable[str]:
        raise NotImplementedError


class LightWord2Vec(BaseEmbedding):
    def __init__(self, model=None) -> None:
        if model is None:
            self._model = fasttext.load_model("fast_text.bin")  # Default model
        else:
            self._model = model

    def __getitem__(self, item: str) -> torch.Tensor:
        return torch.from_numpy(self._model[item])

    def __contains__(self, item: str) -> bool:
        return item in self._model

    def get_vocab(self) -> Iterable[str]:
        return self._model.words
