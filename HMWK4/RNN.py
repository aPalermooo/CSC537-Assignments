##############################################################################
# Name:           RNN.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:
# Date:           27 April 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 4
##############################################################################
import pickle
from typing import Any

import numpy as np
import torch
from numpy import floating
from torch.utils.data import DataLoader

# Global Loss function
loss_function = torch.nn.CrossEntropyLoss()

TRANSLATOR_PACKAGE_PATH = "dataset/translator/translator.pkl"

with open(TRANSLATOR_PACKAGE_PATH, "rb") as f:
    package = pickle.load(f)
    char2idx, idx2char = package


class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, isLSTM: bool = False):
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

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x, h = self.rnn(x, h)
        x = self.fc(x)
        return x, h


def train(
        model: torch.nn.Module,
        training_data: DataLoader,
        validation_data: DataLoader,
        training_tracker: list[Any] = None,
        validation_tracker: list[Any] = None,
        optimizer: torch.optim.Optimizer = None,
        TOTAL_EPOCHS: int = 100,
        THRESH: float = float('-inf'),
        verbose: bool = False,
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(TOTAL_EPOCHS):
        if verbose and epoch % LOG_INTERVAL == 0:  # Verbose Logging
            print(f"\t\tEpoch {epoch + 1}/{TOTAL_EPOCHS}")

        model.train()

        training_loss_per_epoch: list[float] = []

        for inputs, targets in training_data:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device

            optimizer.zero_grad()  # Forward Pass
            hidden = None
            output, hidden = model(inputs, hidden)

            batch_size, seq_len, vocab_size = output.shape  # Calculate Loss
            loss = criterion(output.view(batch_size * seq_len, vocab_size), targets.view(batch_size * seq_len))

            loss.backward()  # Backward Pass

            optimizer.step()  # Update Model Parameters

            training_loss_per_epoch.append(loss.item())

        # Calculate statistics over epoch

        training_tracker.append(np.mean(training_loss_per_epoch))
        avg_loss = evaluate(model, validation_data, device=device)  # Validate
        validation_tracker.append(avg_loss)

        if verbose and epoch % LOG_INTERVAL == 0:  # Verbose Logging
            print(f"\t\t\ttraining avg_loss={training_tracker[-1]:.4f}")
            print(f"\t\t\tvalidation avg_loss={avg_loss:.4f}")

        if abs(prev_loss - avg_loss) < THRESH:  # Early stopping condition
            break
        prev_loss = avg_loss
    return model


def evaluate(
        model: torch.nn.Module,
        validation_data: DataLoader,
        device: torch.device = torch.device("cpu")
) -> floating[Any]:
    """
    Evaluates RNN model on a given data set (No changes made to trainable parameters)

    Args:
        model (torch.nn.Module): RNN model
        validation_data (DataLoader): Data set used to observe performance of RNN model
        device (torch.device, optional): device used for training; defaults to cpu

    Returns:
        Mean loss of model calculated from all evaluation cases
    """

    criterion = loss_function.to(device)
    model = model.to(device)
    loss_tracker = []

    model.eval()

    with torch.no_grad():
        for inputs, targets in validation_data:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device

            hidden = None
            output, hidden = model(inputs, hidden)

            batch_size, seq_len, vocab_size = output.shape  # Calculate Loss
            loss = criterion(output.view(batch_size * seq_len, vocab_size), targets.view(batch_size * seq_len))

            loss_tracker.append(loss.item())
    return np.mean(loss_tracker)


def generate(
        model: torch.nn.Module,
        input: str,
        steps: int = 200,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu")
) -> str:
    """
    Generates a string of text based on previous training and given string

    Args:
        model:
        input:
        steps:
        temperature:
        device:

    Returns:

    """

    model.eval()
    model = model.to(device)

    generated_text = []
    with torch.no_grad():
        tokens = [char2idx[char] for char in input.lower()]  # Prepare Input
        seed = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

        hidden = None
        output, hidden = model(seed, hidden)

        logits = output[:, -1, :] / temperature
        next_char = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

        generated_text.append(idx2char[next_char.item()])

        for _ in range(steps - 1):
            seed = next_char.reshape(1, 1)  # The current output is the NEXT input
            output, hidden = model(seed, hidden)  # Forward pass
            last_step_logits = output[:, -1, :] / temperature  # Use only the last time step
            next_char = torch.multinomial(torch.softmax(last_step_logits, dim=-1),
                                          num_samples=1)  # softmax and then sample on
            generated_text.append(idx2char[next_char.item()])  # append the result
    return "".join(generated_text)


if __name__ == "__main__":
    model = RNN(len(char2idx), 64, len(char2idx))
    print(generate(
        model,
        "The ",
        temperature=1.0,
        device=torch.device("cpu")))
