from nltk.translate.bleu_score import sentence_bleu

REWARD_MULTIPLIER = 100
POSITIVE_REWARD_THRESHOLD = 0.75


def bleu_reward(sent1, sent2):
    return REWARD_MULTIPLIER(sentence_bleu(sent1, sent2) - POSITIVE_REWARD_THRESHOLD)
