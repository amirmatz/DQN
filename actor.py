import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import config
from language import SOS_TOKEN_INDEX, EOS_TOKEN_INDEX


class Actor(nn.Module):

    def __init__(self, embedding_size, hidden_size, output_lang, embedding):
        self.encoder = EncoderRNN(embedding_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_lang.size())
        self.output_lang = output_lang
        self.embedding = embedding

    def forward(self, x, allowed_actions):
        encoder_hidden = self.encoder.init_hidden()
        input_length = len(x)
        encoder_outputs = torch.zeros(config.MAX_LENGTH, self.encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(self.embedding[x[ei]], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN_INDEX]])  # SOS
        decoder_hidden = encoder_hidden
        decoder_attentions = torch.zeros(config.MAX_LENGTH, config.MAX_LENGTH)

        states = [decoder_hidden]
        actions = []

        not_allowed_actions = torch.ones(self.output_lang.size(), 1)
        not_allowed_actions[[self.output_lang.word2index[act] for act in allowed_actions]] = 0

        probs = []

        for di in range(config.MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            states.append(decoder_hidden)
            decoder_attentions[di] = decoder_attention.data
            distribution = decoder_output.data[:]
            distribution[not_allowed_actions] = 0
            distribution = Categorical(logits=distribution)

            action = distribution.sample().detach()
            probs.append(decoder_output.data[action])
            actions.append(action)
            if action == EOS_TOKEN_INDEX: # if we finished the sequence
                break

            decoder_input = action

        actions.append(None)  # last state is terminal and so does not have an action
        probs.append(0)
        return states, actions, probs


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=config.MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)



