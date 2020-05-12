import torch
import torch.nn as nn
from torch.distributions import Categorical
import config


class ActorCopy(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_lang, embedding):
        super(ActorCopy, self).__init__()
        self.encoder = EncoderBiRNN(embedding_size, hidden_size)
        self.decoder = CopyDecoder(output_lang.size(), embedding_size, hidden_size)
        self.output_lang = output_lang
        self.embedding = embedding
        self.hidden_size = hidden_size

    def forward(self, x, allowed_actions):
        encoder_hidden = None
        input_length = len(x)
        encoder_outputs = torch.zeros(config.MAX_LENGTH, self.encoder.hidden_size)

        for ei, word in enumerate(x):
            encoder_output, encoder_hidden = self.encoder(self.embedding[word], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = self.embedding[config.SOS_TOKEN]  # SOS
        decoder_hidden = [item.view(1, 1, -1) for item in encoder_hidden]

        states = [decoder_hidden[0]]
        actions = []

        # allowed actions indices are every allowed_action and every word in sentence except SOS and EOS
        allowed_actions_indices = torch.zeros(self.output_lang.size() + config.MAX_LENGTH)
        allowed_actions_indices[[self.output_lang.word_to_index(act) for act in allowed_actions]] = 1
        allowed_actions_indices[self.output_lang.size() + 1:self.output_lang.size() + input_length - 1] = 1

        probs = []

        prev_word = None
        prev_probs = None
        for di in range(config.MAX_LENGTH):
            # decoder_input
            decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs,
                                                          prev_word, x, prev_probs, decoder_hidden)

            states.append(decoder_hidden[0])
            distribution = decoder_output.squeeze() * allowed_actions_indices
            distribution_caterogical = Categorical(probs=distribution)
            action_idx = distribution_caterogical.sample()
            is_vocab = lambda i: i < self.output_lang.size()
            get_word = lambda i: self.output_lang.index2word[i] if is_vocab(i) else x[i - self.output_lang.size()]
            action = get_word(action_idx)

            prob = distribution[action_idx]
            if is_vocab(action_idx):
                for i, w in enumerate(x):  # If word in vocab check if it is in sentence
                    if w == x:  # Might appear multiple times
                        prob += distribution[self.output_lang.size() + i]
            else:  # If word in sentence then find prob in vocab
                prob += distribution[self.output_lang.word_to_index(action)]

            probs.append(prob)

            actions.append(action)

            if action == config.EOS_TOKEN:  # if we finished the sequence
                break

            prev_word = actions[-1]
            prev_probs = decoder_output.detach()
            decoder_input = self.embedding[action]  # oov index if not used

        return states, actions, probs


class EncoderBiRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(EncoderBiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size // 2, num_layers=1, bidirectional=True)

    def forward(self, input: torch.Tensor, hidden):
        output, hidden = self.lstm(input.view(1, 1, input.size(0)), hidden)
        return output, hidden


class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_length=config.MAX_LENGTH):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=embed_size + hidden_size * 2, hidden_size=hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # weights
        self._generate_layer = nn.Linear(hidden_size, vocab_size)  # generate mode
        self._copy_layer = nn.Linear(hidden_size, hidden_size)  # copy mode

    def forward(self, input, encoder_outputs, prev_word, sentence, prev_probs, hidden):
        if prev_word is None:
            selective_read = torch.zeros(self.hidden_size)
            attentive_read = torch.zeros(self.hidden_size)
        else:
            a = torch.cat([input, hidden[0].squeeze()]).unsqueeze(0)
            b = self.attn(a)
            attentive_read = torch.softmax(b, dim=1)
            attentive_read = torch.matmul(attentive_read, encoder_outputs).squeeze()  # [hidden_size]
            # TODO Amir: is this the way you visioned it? Why not use an implementation of attention?

            probs_c = self._get_copy_probs(sentence, prev_probs, prev_word)
            selective_read = torch.matmul(probs_c.unsqueeze(0), encoder_outputs).squeeze()  # [hidden_size]

        # 1. update states
        a = torch.cat([input, selective_read, attentive_read]).unsqueeze(0).unsqueeze(0)  # [1 x1 x (h*2 + emb)]
        _, hidden = self.lstm(a, hidden)  # s_t = f(y_(t-1), s_t-1, c_t)

        generate_score = self._get_generate_score(hidden[0])

        # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
        copy_score = self._get_copy_score(encoder_outputs, hidden[0])

        # after this section - section c is done

        # ....
        # 2-3) get softmax-ed probabilities
        score = torch.cat([generate_score, copy_score], dim=2)  # [(vocab+seq)]
        probs = torch.softmax(score, dim=2)
        return probs, hidden

    def _get_copy_score(self, encoder_outputs, hi):
        copy_proj = self._copy_layer(encoder_outputs)  # [seq x hidden_size]
        score_c = torch.tanh(copy_proj)  # [seq x hidden_size]
        return torch.matmul(score_c, hi.squeeze()).unsqueeze(0).unsqueeze(0)  # [seq]

    def _get_generate_score(self, hi):
        # 2. predict next word y_t
        # 2-1) get scores score_g for generation- mode
        return self._generate_layer(hi)  # [1 x vocab_size]

    def _get_copy_probs(self, sentence, prev_probs, prev_word):
        probs_c = prev_probs.squeeze()[self.vocab_size:]

        unused_words = torch.ones(probs_c.shape)
        unused_words[0] = 0  # SOS
        unused_words[len(sentence) - 1:] = 0  # EOS until the end
        unused_words[[idx for idx in range(1, len(sentence)) if sentence[idx] == prev_word]] = 0  # The prev word
        # TODO Amir: this was `sentence[idx] != prev_word` it was a bug right?

        probs_c *= unused_words
        probs_sum = probs_c.sum()
        if probs_sum > 0:
            probs_c /= probs_sum

        return probs_c
