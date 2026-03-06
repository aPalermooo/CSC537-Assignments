##############################################################################
# Name:           mlp.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:
# Date:           6 March 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 2
##############################################################################
from unittest import case

import torch.nn


class MLP (torch.nn.Module):

    # https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    def __init__(self, input_dim : int, hidden_size : int, hidden_dim : int, activation : str, output_dim : int):
        """
        Creates a configurable MultiLayer Perceptron model.
        :param input_dim: the number of input features ( int > 0 )
        :param hidden_size: the number of hidden layers in the model ( int > 0 )
        :param hidden_dim: the number of neurons in each hidden layer ( int > 0 )
        :param activation: the activation function used for the hidden layers.
                            **precondition:** must be `relu`, `sigmoid`, or `tanh`
        :param output_dim: the number of output neurons ( int > 0 )
        :except ValueError param given to construct model is not valid
        """
        super(MLP, self).__init__()
        self.flatten = torch.nn.Flatten()

        ## Construct layers

        layers = []

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_size = output_dim

        # validate inputs
        match activation.lower():
            case "relu":
                self.activation = torch.nn.ReLU()
            case "sigmoid":
                self.activation = torch.nn.Sigmoid()
            case "tanh":
                self.activation = torch.nn.Tanh()
            case _:
                raise ValueError("Unknown activation function")

        if input_dim <= 0:
            raise ValueError("Invalid input dimension; there must be at least 1 dimension")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0; there must be at least 1 hidden layer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0; there must be at least 1 neuron in the layer")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0; there must be at least 1 output")

        # dynamically generate layers
        layers.append(torch.nn.Linear(in_features=input_dim, out_features=hidden_dim))         # input layer + first hidden layer
        layers.append(self.activation)

        for k in range(1, hidden_size):                                                          # k-1 more hidden layers
            layers.append(torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(self.activation)

        layers.append(torch.nn.Linear(in_features=hidden_dim, out_features=output_dim))        # output layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Takes input tensor, and simulates MLP to generate an output tensor
        :param x: input features
        :return: generated output by MLP
        :except ValueError number of features given does not match the number of input features
        """
        x = self.flatten(x)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input must have {self.input_dim} features")
        return self.model(x)