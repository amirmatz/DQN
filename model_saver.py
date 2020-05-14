import pickle

import config
from actor_copy import ActorCopy
from language import Lang
from vectorize_words import LightWord2Vec

try:
    from google.colab import files

    IS_COLAB = True
except:
    IS_COLAB = False


def save_model(epoch, actor, critic, critic_optimizer, critic_criterion,
               actor_optimizer, lang):
    torches = [critic, critic_optimizer, critic_criterion, actor_optimizer, actor.encoder, actor.decoder,
               lang.index2word]
    with open(f"pickles/epoch_{epoch}.pkl", "wb") as f:
        pickle.dump(torches, f)

    if IS_COLAB:
        files.download(f"pickles/epoch_{epoch}.pkl")


def load_model(epoch):
    with open(f"pickles/epoch_{epoch}.pkl", "rb") as f:
        pickled = pickle.load(f)

    word2vec = LightWord2Vec()
    lang = Lang(word2vec.get_vocab())
    actor = ActorCopy(config.EMBEDDING_SIZE, config.STATE_SIZE, lang, word2vec)

    lang.index2word = pickled.pop()
    for index, word in enumerate(lang.index2word):
        lang.word2index[word] = index

    actor.decoder = pickled.pop()
    actor.encoder = pickled.pop()

    return [actor] + pickled + [lang]
