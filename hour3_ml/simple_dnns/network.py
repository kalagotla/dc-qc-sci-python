# Shared neural network module for simple_dnns package

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    A simple feedforward neural network with configurable layers and activation.
    
    Parameters
    ----------
    layers : list
        List of integers specifying the size of each layer.
        Example: [1, 5, 10, 5, 1] creates a network with 1 input, 
        three hidden layers (5, 10, 5 neurons), and 1 output.
    activation : str, optional
        Activation function to use ('tanh' or 'relu'). Default is 'tanh'.
    """

    def __init__(self, layers, activation='tanh'):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        self.activation = activation
        
        for input_size, output_size in zip(layers, layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                if self.activation == 'relu':
                    x = torch.relu(linear_transform(x))
                else:
                    x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x
