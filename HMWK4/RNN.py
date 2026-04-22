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
from typing import Any

import numpy as np
import torch
from numpy import floating
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
    LOG_INTERVAL: int = 10,
    device: torch.device = torch.device("cpu")
    ) -> torch.nn.Module:
    """
    Trains RNN model on a given data set

    Training Tracking and Validation tracking are done in place on the parameters passed into the function

    Precondition:
        Training data and Validation data are different sets

    Args:
        model (torch.nn.Module): RNN model
        training_data (DataLoader, optional): Data set used to train the model
        validation_data (DataLoader, optional): Data set used to observe how model does on previously unseen inputs
        training_tracker (list[Any], optional): Empty list used to preserve data collected on tracking training of model
        validation_tracker (list[Any], optional): Empty list used to preserve data collected on tracking validation of model
        optimizer (torch.optim.Optimizer, optional): optimizer for training; Defaults to ADAM
        TOTAL_EPOCHS (int, optional): number of epochs before training is forced stopped
        THRESH (float, optional): threshold for early stopping condition; defaults to Off
        verbose (bool, optional): toggles verbose logging during training; defaults to False (Off)
        LOG_INTERVAL (int, optional): How often verbose logging is done; defaults to every 10 epochs
        device (torch.device, optional): device used for training; defaults to cpu

    Returns:
        Model trained on data until convergence is found
    """

    # global vars
    prev_loss = 100.

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
        if verbose and epoch % LOG_INTERVAL == 0:                       # Verbose Logging
            print(f"\t\tEpoch {epoch+1}/{TOTAL_EPOCHS}")

        model.train()

        training_loss_per_epoch: list[float] = []

        for inputs, targets in training_data:
            inputs, targets = inputs.to(device), targets.to(device)     # Move data to device

            optimizer.zero_grad()                                       # Forward Pass
            output = model(inputs)

            loss = criterion(output, targets)                           # Calculate Loss

            loss.backward()                                             # Backward Pass

            optimizer.step()                                            # Update Model Parameters

            training_loss_per_epoch.append(loss.item())

        # Calculate statistics over epoch

        training_tracker.append(np.mean(training_loss_per_epoch))
        avg_loss = evaluate(model, validation_data, validation_tracker, device=device)  # Validate

        if verbose and epoch % LOG_INTERVAL == 0:                       # Verbose Logging
            print(f"\t\t\ttraining avg_loss={training_tracker[-1]:.4f}")
            print(f"\t\t\tvalidation avg_loss={avg_loss:.4f}")

        if abs(prev_loss - avg_loss) < THRESH:                          # Early stopping condition
            break
        prev_loss = avg_loss
    return model


def evaluate(
    model: torch.nn.Module,
    validation_data : DataLoader,
    validation_tracker : list[Any] = None,
    device : torch.device = torch.device("cpu")
    ) -> floating[Any]:
    """
    Evaluates RNN model on a given data set (No changes made to trainable parameters)

    Args:
        model (torch.nn.Module): RNN model
        validation_data (DataLoader): Data set used to observe performance of RNN model
        validation_tracker (list[Any], optional): empty list used to preserve data collected on tracking evaluation of model
        device (torch.device, optional): device used for training; defaults to cpu

    Returns:
        Mean loss of model calculated from all evaluation cases
    """

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