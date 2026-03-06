##############################################################################
# Name:           MLP.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Creates a configurable model a MultiLayer Perception Model to used to conduct experiments
# Date:           6 March 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 2
##############################################################################
import datetime

import numpy as np
import torch.nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
        layers.append(torch.nn.Linear(in_features=input_dim, out_features=hidden_dim))          # input layer + first hidden layer
        layers.append(self.activation)

        for k in range(1, hidden_size):                                                         # k-1 more hidden layers
            layers.append(torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(self.activation)

        layers.append(torch.nn.Linear(in_features=hidden_dim, out_features=output_dim))         # output layer
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

    def predict(self, x : torch.Tensor) -> torch.Tensor:
        """
        Takes input tensor, and simulates MLP to generate an inference
        :param x: input features
        :return: the inference output
        :except ValueError number of features given does not match the number of input features
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self):
        """
        :return: Reports the number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train(
        model : torch.nn.Module,
        data : DataLoader,
        optimizer : torch.optim.Optimizer,
        is_classification : bool,
        l2 : float = 0.,
        TOTAL_EPOCHS : int = 1000,
) -> tuple[Module, list[float]]:
    """
    Trains a model on a given set of data given a set of customizable parameters
    :param model: The model to be trained
    :type model: torch.nn.Module
    :param data: The features and target values to be used for training
    :type data: DataLoader
    :param optimizer: the torch optimizer to be used for training
    :type optimizer: torch.optim.Optimizer
    :param is_classification: True if model is simulating a classification problem.
                                False if is simulation a regression problem.
    :param l2: Amount of weight decay to be applied to weights in each epoch
    :param TOTAL_EPOCHS: Number of epoches to train the data (no early stop)
    :return: the trained model and a list of avg. loss for each epoch
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{timestamp}")

    loss_tracker = []

    if is_classification:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:   #is Regression Model
        loss_fn = torch.nn.MSELoss()


    # per epoch
    for epoch in range(TOTAL_EPOCHS):
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
        model.train()

        loss_per_epoch : list[float] = []

        # per batch
        for inputs, class_targets, reg_targets in data:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, class_targets.long() if is_classification else reg_targets.unsqueeze(1))
            if l2 != 0.:
                # l2 is enabled
                decay = sum(p.pow(2).sum() for p in model.parameters())
                loss += l2 * decay

            loss.backward()

            optimizer.step()

            loss_per_epoch.append(loss.item())

        writer.add_scalar("Loss/train", np.mean(loss_per_epoch), epoch)
        loss_tracker.append(np.mean(loss_per_epoch))

    writer.close()
    return model, loss_tracker


def evaluate(
        model: torch.nn.Module,
        data: DataLoader,
        is_classification: bool,
) -> tuple[list[float], float] | list[float]:
    """
    Evaluates a given model on its performance on unseen data
    :param model: The model to be evaluated
    :type model: torch.nn.Module
    :param data: The features and target values to be used for evaluation
    :pre: Data has not been yet used to train the model
    :type data: DataLoader
    :param is_classification: True if model is simulating a classification problem.
                                False if is simulation a regression problem.
    :return: The calculated loss of each sample. (if is classification problem, returns accuracy as well)
    """
    model.eval()

    loss_tracker : list[float] = []
    correct = 0
    total = 0

    if is_classification:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:   #is Regression Model
        loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for inputs, class_targets, reg_targets in data:
            outputs = model(inputs)
            loss = loss_fn(outputs, class_targets.long() if is_classification else reg_targets.unsqueeze(1))
            loss_tracker.append(loss.item())

            if is_classification:       #save statistics for accuracy evaluation
                prediction = torch.argmax(outputs, dim=1)
                correct += (prediction == class_targets.long()).sum().item()
                total += class_targets.size(0)

    if is_classification:
        accuracy = correct / total
        return loss_tracker, accuracy
    else:
        return loss_tracker