from collections import deque, namedtuple
import itertools
import random
import torch

import config
from actor import Actor
from critic import Critic
from dataset_reader import DataSetReader
from language import Lang
from reward import bleu_reward
from vectorize_words import LightWord2Vec

Experience = namedtuple('Experience', ['state', 'action', 'next_state', 'reward', 'probs', 'sentence'])


def train():
    experiences_buffer = deque(maxlen=config.MAX_EXPERIENCES_SIZE)
    word2vec = LightWord2Vec()
    lang = Lang(word2vec.get_vocab())
    actor = Actor(config.EMBEDDING_SIZE, config.STATE_SIZE, lang, word2vec)
    critic = Critic(config.STATE_SIZE, config.EMBEDDING_SIZE, config.CRITIC_HIDDEN_SIZE)
    reader = DataSetReader('train')
    critic_optimizer = torch.optim.Adam(critic.parameters())
    critic_criterion = torch.nn.MSELoss()
    actor_optimizer = torch.optim.Adam(actor.parameters())

    for _ in range(config.EPOCHS):

        # training actor
        for x, y in reader.read(config.TRAIN_BATCH_SIZE):
            for sentence, target_sentence in zip(x, y):
                states, actions, probs = actor(sentence, get_possible_actions(lang, sentence))
                predicted_sentence = [lang.index2word[int(action)] for action in actions[:-1]]  # Skip None

                # todo think maybe about a better reward function
                rewards = [bleu_reward(target_sentence[:i + 1], predicted_sentence[:i + 1]) for i in
                           range(max(len(target_sentence), len(predicted_sentence)))] + [0]

                for i in range(len(states) - 1):
                    experiences_buffer.insert(0,
                                              Experience(states[i], actions[i], states[i + 1], rewards[i], probs[i],
                                                         sentence))

        q_estimated = torch.zeros(config.Q_BATCH_SIZE, 1)
        q_s = torch.zeros(config.Q_BATCH_SIZE, 1)

        # training q function
        exp_length = min(len(experiences_buffer), config.Q_BATCH_SIZE)

        for idx in range(exp_length):
            exp = experiences_buffer[random.randint(0, exp_length)]
            action_emb = word2vec[lang.index2word[int(exp.action)]]
            q_estimated[idx] = critic(exp.state, action_emb)
            q_s[idx] = exp.reward
            if exp.next_state is not None:
                with torch.no_grad():
                    q_s[idx] += (config.GAMMA * max([critic(exp.next_state, word2vec[action]) for action in
                                                    get_possible_actions(lang, exp.sentence)]))[0][0]

        critic_optimizer.zero_grad()
        loss = critic_criterion(q_s, q_estimated)
        loss.backward()
        critic_optimizer.step()

        # updating seq2seq model
        actor_optimizer.zero_grad()
        loss = shared_loss(experiences_buffer, q_estimated)
        loss.backward()
        actor_optimizer.step()

        experiences_buffer.clear()


def shared_loss(experience_buffer, q_estimated):
    probs = torch.zeros(len(experience_buffer), 1, dtype=torch.float64)
    for i in range(len(experience_buffer)):
        probs[i] = experience_buffer[i].probs

    return torch.div(torch.sum(torch.mul(probs, q_estimated)), config.TRAIN_BATCH_SIZE)


def get_possible_actions(lang, sentence):
    return itertools.chain(sentence, lang.get_actions())


# todo
# set encoding from word2vec
# set data reader
# check q function recommended layers


if __name__ == '__main__':
    train()
