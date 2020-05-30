import os
import sys

import pandas as pd
from dataset_reader import DataSetReader
from model_saver import load_model
from train import get_possible_actions


def lists_to_sentances(*lst):
    return [" ".join(l) for l in lst]


def generate_output(actor, lang, mode: str) -> pd.DataFrame:
    result = []
    for input_sentence, expected_qdmr in DataSetReader(mode).get_all():
        states, actions, probs = actor(input_sentence, get_possible_actions(lang, input_sentence))
        predicted_sentence = [action for action in actions[:-1]]
        result.append(lists_to_sentances(input_sentence, expected_qdmr, predicted_sentence))

    return pd.DataFrame(result, columns=["input", "expected", "actual"])


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError(f"Insert epoch and mode as arg: {os.path.basename(__file__)} <epoch_number> <train/test>")

    epoch = int(sys.argv[1])
    mode = sys.argv[2]
    actor, critic, critic_optimizer, critic_criterion, actor_optimizer, lang = load_model(epoch)
    result = generate_output(actor, lang, mode)
    result.to_pickle(f"result_{mode}_{epoch}.pkl", compression="gzip")
