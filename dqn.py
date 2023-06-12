import torch.nn as nn
import torch


class DQN(nn.Module):
    # this is the network for the Q-function 
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        # super initial all the parent objects
        super(DQN, self).__init__()
        # create two layers for the input->h, and h->output
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        # use relu as activation 
        self.relu = nn.ReLU()
        # design the dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # first layer: linear + dropout + relu
        out = self.input_to_hidden(x)
        out = self.dropout(out)
        out = self.relu(out)
        # second layer: linear + dropout
        # the output will be a array for each card
        # the logit will be calculate after the modeling for the softmax
        out = self.hidden_to_output(out)
        out = self.dropout(out)
        return out
