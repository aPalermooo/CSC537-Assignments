##############################################################################
# Name:           SimpleRNN.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:
# Date:           27 April 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 4
##############################################################################
from platform import architecture
from typing import Any

import numpy as np
import torch
from numpy import floating
from torch.nn import Module
from torch.utils.data import DataLoader

# Global Loss function
loss_function = torch.nn.CrossEntropyLoss()

class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, isLSTM : bool = False):
        super(RNN, self).__init__()

        self.isLSTM = isLSTM

        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        if isLSTM:
            self.rnn = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = torch.nn.RNN(hidden_size, hidden_size, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def train(
    model: torch.nn.Module,
    training_data : DataLoader,
    validation_data : DataLoader,
    training_tracker : list[Any] = None,
    validation_tracker : list[Any] = None,
    optimizer: torch.optim.Optimizer = None,
    TOTAL_EPOCHS: int = 100,
    THRESH : float = float('-inf'),
    verbose : bool = False,
    device: torch.device = torch.device("cpu")
    ):

    # Move objects to device
    model = model.to(device)
    criterion = loss_function.to(device)

    if training_tracker is None:
        training_tracker = list()

    if validation_tracker is None:
        validation_tracker = list()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(TOTAL_EPOCHS):
        if verbose and epoch % 10 == 0:                    # Verbose Logging
            print(f"\t\tEpoch {epoch+1}/{TOTAL_EPOCHS}")

        model.train()

        for inputs, targets in training_data:
            inputs, targets = inputs.to(device), targets.to(device)     # Move data to device

            optimizer.zero_grad()                                       # Forward Pass
            output = model(inputs)

            loss = criterion(output, targets)                           # Calculate Loss

            loss.backward()                                             # Backward Pass

            optimizer.step()                                            # Update Model Parameters

    return model

def evaluate(
    model: torch.nn.Module,
    validation_data : DataLoader,
    validation_tracker : list[Any] = None,
    device : torch.device = torch.device("cpu")
    ) -> floating[Any]:

    if validation_tracker is None:
        validation_tracker = list()

    criterion = loss_function.to(device)
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        for inputs, targets in validation_data:
            inputs, targets = inputs.to(device), targets.to(device)         # Move data to device

            output = model(inputs)                                          # Forward Pass

            loss = criterion(output, targets)                               # Calculate Loss

            validation_tracker.append(loss.item())
    return np.mean(validation_tracker)