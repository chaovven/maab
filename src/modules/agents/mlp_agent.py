import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)  # input_size, hidden_size
        self.fc3 = nn.Linear(args.rnn_hidden_dim, output_shape)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))  # hidden_states
        q = self.fc3(x)  # agent_outs
        hidden = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return q, hidden
