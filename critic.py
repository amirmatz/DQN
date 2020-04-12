import torch

HIDDEN_LAYER_SIZE = 100


class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(action_size, hidden_size)
        self.fc3 = torch.nn.Linear(2*hidden_size, 1)

    def forward(self, state, action):
        state_out = torch.nn.functional.relu(self.fc1(state))
        action_out = torch.nn.functional.relu(self.fc2(action))
        out = torch.nn.functional.relu(torch.cat((state_out, action_out)))
        out = self.fc3(out)
        return out


