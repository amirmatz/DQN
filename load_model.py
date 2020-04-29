from actor import Actor
from dataset_reader import wrap_sentence
from model_saver import load_model
from train import get_possible_actions

actor, critic, critic_optimizer, critic_criterion, actor_optimizer, lang = load_model(0)


def test_actor(actor: Actor, sentence: str) -> None:
    sentence = wrap_sentence(sentence)
    states, actions, probs = actor(sentence, get_possible_actions(lang, sentence))
    predicted_sentence = [lang.index2word[int(action)] for action in actions[:-1]]
    print(predicted_sentence)


test_actor(actor, "return me the homepage of PVLDB . ")
