import re
from typing import Iterable, Tuple

import pandas as pd

import config
from language import Lang

_mode_to_files = {
    "train": [],  # TODO: insert desired input path
    "test": []
}


def wrap_sentences(sentences: Iterable[str]) -> Iterable[str]:
    for sent in sentences:
        yield f"{config.SOS_TOKEN} {sent} {config.EOS_TOKEN}"


def fix_references(string: str) -> str:
    return re.sub(r'#([1-9][0-9]?)', '@@\g<1>@@', string)


def process_target(target: str) -> str:
    # replace multiple whitespaces with a single whitespace.
    target_new = ' '.join(target.split())

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = target_new.split(';')
    new_parts = [re.sub(r'return', '', part.strip()) for part in parts]
    target_new = ' @@SEP@@ '.join([part.strip() for part in new_parts])

    # replacing references with special tokens, for example replacing #2 with @@2@@.
    target_new = fix_references(target_new)

    return target_new.strip()


def _load_mode_df(mode) -> pd.DataFrame:
    file_paths = _mode_to_files[mode]
    dfs = []
    for file_path in file_paths:
        dfs.append(pd.read_csv(file_path))

    return pd.concat(dfs, ignore_index=True, sort=False)


class DataSetReader:
    def __init__(self, mode: str, lang: Lang) -> None:
        self._file_name = _load_mode_df(mode)
        self._orig_df = pd.read_csv(self._file_name)
        # TODO: run lang on input?

    def read(self, batch_size) -> Iterable[Tuple[str, str]]:
        sample = self._orig_df.sample(batch_size)

        x = sample["question_text"]
        y = sample["decomposition"].apply(process_target)
        return zip(wrap_sentences(x), wrap_sentences(y))
