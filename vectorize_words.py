import gzip
import json
from typing import Iterable

import numpy as np


class Word2Vec:
    def __init__(self) -> None:
        with gzip.open("word2vec.json.gz") as f:
            self._d = json.load(f)

        for k, v in self._d.items():
            self._d[k] = np.array(v)

    def __getitem__(self, item):
        if item in self._d:
            return self._d[item]

        raise NotImplementedError("What to do if not found?")  # TODO: implement

    def get_vocab(self) -> Iterable[str]:
        return self._d.keys()