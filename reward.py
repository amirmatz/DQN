from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sari import SARIsent

REWARD_MULTIPLIER = 100
POSITIVE_REWARD_THRESHOLD = 0.75


def list_to_sentence(lst):
    return " ".join(lst)


def bleu_reward(sent1, sent2):
    cc = SmoothingFunction()
    return REWARD_MULTIPLIER * (
            sentence_bleu([sent1], sent2, smoothing_function=cc.method3) - POSITIVE_REWARD_THRESHOLD)


def sari_reward(orig_sentence, predict_sentence, ref_sentence):
    return SARIsent(list_to_sentence(orig_sentence), list_to_sentence(predict_sentence),
                    [list_to_sentence(ref_sentence)])
