import pickle

import config
from actor import Actor
from dataset_reader import wrap_sentence
from language import Lang
from train import get_possible_actions
from vectorize_words import LightWord2Vec

word2vec = LightWord2Vec()
lang = Lang(word2vec.get_vocab())
actor = Actor(config.EMBEDDING_SIZE, config.STATE_SIZE, lang, word2vec)

with open(f"pickles/epoch_550.pkl", "rb") as f:
    actor.encoder, actor.decoder, lang.index2word, critic, critic_optimizer, critic_criterion, actor_optimizer = pickle.load(
        f)

for index, word in lang.index2word.items():
    lang.word2index[word] = index


def test_actor(actor: Actor, sentence: str) -> None:
    sentence = wrap_sentence(sentence)
    states, actions, probs = actor(sentence, get_possible_actions(lang, sentence))
    predicted_sentence = [lang.index2word[int(action)] for action in actions[:-1]]
    print(predicted_sentence)


test_actor(actor, "return me the homepage of PVLDB . ")
