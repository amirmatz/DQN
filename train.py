from collections import deque, namedtuple
import random
import torch

import config
from actor import Actor
from critic import Critic
from dataset_reader import DataSetReader
from language import Lang
from reward import bleu_reward

Experience = namedtuple('Experience', ['state', 'action', 'new_state', 'reward', 'probs'])


def train():
    experiences_buffer = deque(maxlen=config.MAX_EXPERIENCES_SIZE)
    lang = Lang()
    actor = Actor(config.EMBEDDING_SIZE, config.STATE_SIZE, lang)
    critic = Critic(config.STATE_SIZE, config.EMBEDDING_SIZE, config.CRITIC_HIDDEN_SIZE)
    reader = DataSetReader('train', lang)
    critic_optimizer = torch.optim.Adam(critic.parameters())
    critic_criterion = torch.nn.MSELoss()
    actor_optimizer = torch.optim.Adam(actor.parameters())

    for _ in range(config.EPOCHS):

        # training actor
        for x, y in reader.read(config.TRAIN_BATCH_SIZE):
            for sentence, target_sentence in zip(x, y):
                states, actions, probs = actor(sentence)
                predicted_sentence = [lang.index2word[action] for action in actions]

                # todo think maybe about a better reward function
                rewards = [bleu_reward(target_sentence[:i+1], target_sentence[:i+1]) for i in range(len(predicted_sentence))]

                for i in range(len(sentence)):
                    if i == len(sentence) - 1:
                        experiences_buffer.insert(0, Experience(states[i], actions[i], None, rewards[i], probs[i]))
                    else:
                        experiences_buffer.insert(0, Experience(states[i], actions[i], states[i+1], rewards[i], probs[i]))

        q_estimated = torch.zeros(config.Q_BATCH_SIZE, 1)
        q_s = torch.zeros(config.Q_BATCH_SIZE, 1)

        # training q function
        exp_length = min(len(experiences_buffer), config.Q_BATCH_SIZE)

        for idx in range(exp_length):
            exp = experiences_buffer[random.randint(exp_length)]
            q_estimated[idx] = critic(exp.state, exp.action)
            q_s[idx] = exp.reward
            if exp.next_state is not None:
                with torch.no_grad():
                    embedding = actor.encoder.embedding
                    q_s[idx] += config.GAMMA * max([critic(exp.next_state, action) for action in get_possible_actions(lang, embedding)])

        critic_optimizer.zero_grad()
        loss = critic_criterion(q_s, q_estimated)
        loss.backward()
        critic_optimizer.step()


        # updating seq2seq model
        actor.zero_grad()
        actor_optimizer.zero_grad()
        loss = shared_loss(experiences_buffer, q_estimated)
        loss.backward()
        actor_optimizer.step()

        experiences_buffer.clear()


def shared_loss(experience_buffer, q_estimated):
    probs = torch.zeros(len(experience_buffer), 1, dtype=torch.float64)
    for i in range(len(experience_buffer)):
        probs[i] = experience_buffer[i][2].probs

    return torch.div(torch.sum(torch.mul(probs, q_estimated)), config.TRAIN_BATCH_SIZE)


def get_possible_actions(lang, embedding):
    with torch.no_grad():
        return (embedding(word) for word in lang.words())

# todo
# set encoding from word2vec
# set data reader
# check q function recommended layers


if __name__ == '__main__':
    train()
