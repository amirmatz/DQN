import pickle
from io import StringIO

import torch

from language import Lang

try:
    from google.colab import files

    IS_COLAB = True
except:
    IS_COLAB = False


def save_model(epoch, actor, critic, critic_optimizer, critic_criterion,
               actor_optimizer, lang):
    torches = [actor, critic, critic_optimizer, critic_criterion, actor_optimizer]
    pickled = []
    for torch_obj in torches:
        s = StringIO()
        torch.save(torch_obj, s)
        pickled.append(s)

    pickled.append(lang.index2word)
    with open(f"pickles/epoch_{epoch}.pkl", "wb") as f:
        pickle.dump(pickled, f)

    if IS_COLAB:
        files.download(f"pickles/epoch_{epoch}.pkl")


def load_model(epoch):
    with open(f"pickles/epoch_{epoch}.pkl", "rb") as f:
        pickled = pickle.load(f)

    lang = Lang([])
    lang.index2word = pickled.pop()
    for index, word in lang.index2word.items():
        lang.word2index[word] = index

    output = []
    for torch_obj in pickled:
        output.append(torch.load(torch_obj))

    output.append(lang)

    return output
