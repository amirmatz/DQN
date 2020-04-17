import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
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

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(self.embedding[x[ei]], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = self.embedding(config.SOS_TOKEN)  # SOS
        decoder_hidden = encoder_hidden

        states = [decoder_hidden]
        actions = []

        not_allowed_actions = np.ones(self.output_lang.size()+len(actions))
        not_allowed_actions[[self.output_lang.word_to_index(act) for act in allowed_actions]] = 0
        not_allowed_actions[self.output_lang.size():self.output_lang.size()+len(x)] = 0

        probs = []

        prev_word = None
        prev_probs = None
        for di in range(config.MAX_LENGTH):
            # decoder_input
            decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs,
                                                                    prev_word, x, prev_probs, decoder_hidden)

            states.append(decoder_hidden)

            distribution = decoder_output.data[:]
            distribution[0][not_allowed_actions] = 0
            distribution = Categorical(logits=distribution)

            # todo update Actor-Critic Model to expect action word instead of embedding

            action = distribution.sample().detach()
            is_vocab = lambda i: i < self.output_lang.size()
            get_word = lambda i: self.output_lang.index2word[i] if is_vocab(i) else x[i-self.output_lang.size()]
            action = get_word(action)

            probs.append(sum([decoder_output.data[0][idx] for idx in range(len(decoder_output.data[0]))
                              if get_word(idx) == action]))

            actions.append(action)

            if action == config.EOS_TOKEN:  # if we finished the sequence
                break

            prev_word = actions[-1]
            prev_probs = probs[-1]
            decoder_input = self.embedding(action) # oov index if not used

        actions.append(None)  # last state is terminal and so does not have an action
        probs.append(0)
        return states, actions, probs


# todo convert Encoder to bidrectional
class EncoderBiRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(EncoderBiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size/2, num_layers=1, bidirectional=True)

    def forward(self, input: torch.Tensor, hidden):
        output, hidden = self.lstm(input.view(1, 1, input.size(0)), hidden)
        return output, hidden

    def init_hidden(self):  # TODO: I think we can delete this
        return torch.zeros(1, 1, self.hidden_size)


class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=embed_size+hidden_size*2, hidden_size=hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # weights
        self.Ws = nn.Linear(hidden_size*2, hidden_size) # only used at initial stage
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2, hidden_size) # copy mode

    def forward(self, input, encoder_outputs, prev_word, sentence, prev_probs, hidden):
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        if prev_word is None:
            selective_read = torch.zeros(1, self.hidden_size)
            attentive_read = torch.zeros(1, self.hidden_size)
        else:
            a = torch.cat((input[0], hidden[0]), 1).reshape(1,1,-1)
            b = self.attn(a)
            attentive_read = F.softmax(b, dim=1)
            probs_c = prev_probs[vocab_size:]
            selective_read = probs_c * encoder_outputs
            selective_read = selective_read[[idx for idx in range(len(sentence)) if sentence[idx] == prev_word]]\
                               / selective_read.sum()

        # 1. update states
        a = torch.cat([input, selective_read, attentive_read], 1).reshape(1,1,-1) # 1 * (h*2 + emb)
        _, hidden = self.lstm(a, hidden) # s_t = f(y_(t-1), s_t-1, c_t)

        # 2. predict next word y_t
        # 2-1) get scores score_g for generation- mode
        score_g = self.Wo(hidden) # [vocab_size]

        # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
        score_c = F.tanh(self.Wc(encoder_outputs.view(-1,hidden_size*2))) # [1*seq x hidden_size]
        score_c = score_c.view(1, -1, hidden_size) # [1 x seq x hidden_size]
        score_c = torch.bmm(score_c, hidden.unsqueeze(2)).squeeze() # [1 x seq]

        # after this section - section c is done

        # ....
        # 2-3) get softmax-ed probabilities
        score = torch.cat([score_g, score_c], 1) # [1 x (vocab+seq)]
        probs = F.softmax(score)
        return probs, hidden
