##############################################################################
# Name:           cnn.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Custom class implementation of a simple CNN model used to conduct experiments
# Date:           3 April 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 3
##############################################################################
from typing import Any

import numpy as np
import torch
from numpy import floating
from torch.nn import Module
from torch.utils.data import DataLoader

# Global Loss function (for classification)
loss_function = torch.nn.CrossEntropyLoss()

class SimpleCNN(torch.nn.Module):

    def __init__(self, depth=2):
        """
        A simple CNN model with 3 convolutional modules of a customizable depth.
        Uses Global Average Pooling and an FC layer to classify images.

        Args:
            depth (int, optional): Depth of the CNN. Defaults to 2.
        """
        super(SimpleCNN, self).__init__()

        self.module1 = self.__make_module(in_channels=3, out_channels=32, depth=depth)

        self.module2 = self.__make_module(in_channels=32, out_channels=64, depth=depth, stride=2)

        self.module3 = self.__make_module(in_channels=64, out_channels=128, depth=depth, stride=2)

        self.gap = torch.nn.AdaptiveAvgPool2d(1)

        self.fc = torch.nn.Linear(in_features=128, out_features=10)

        self.__init_weights()

    def __make_module(
            self,
            in_channels : int,
            out_channels : int,
            depth : int,
            kernel_size : int =3,
            padding : int =1,
            stride : int =1
            ) -> torch.nn.Sequential:
        """
        Private Function

        Dynamically assembles one CNN module of a desired depth.

        (Conv + BatchNorm + ReLU) x depth

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            depth (int): Depth of the CNN.
            kernel_size (int, optional): Kernel size of the CNN. Defaults to 3.
            padding (int, optional): Padding of the CNN. Defaults to 1.
            stride (int, optional): Stride of the CNN. Defaults to 1.

        Returns:
            torch.nn.Sequential: Assembled CNN module
        """
        layers = []

        for i in range(depth):
            layers.append(
                torch.nn.Conv2d(in_channels if i == 0 else out_channels,
                                out_channels, kernel_size, padding=padding,
                                stride=stride if i == 0 else 1),
            )
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers)

    def __init_weights(self) -> None:
        """
        Private Function

        Initializes weights of all trainable parameters in the CNN.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward Propagation through the network

        Args:
            x (torch.Tensor): Input features

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train (
        model : torch.nn.Module,
        training_data : DataLoader,
        validation_data : DataLoader,
        optimizer : torch.optim.Optimizer = None,
        TOTAL_EPOCHES : int = 100,
        THRESH : float = float('-inf'),
        verbose : bool = False,
        device: torch.device = torch.device('cpu')
        ) -> tuple[Module, list[Any], list[Any], list[Any]]:
    """
    Trains CNN model on a given set of data

    Args:
        model (torch.nn.Module) : The model to be trained
        training_data (DataLoader) : Data used to train the model
        validation_data (DataLoader) : Data used to validate the model after every epoch
        optimizer (torch.optim.Optimizer, optional) : Optimizer used to train the model. Defaults to Adam with lr of 0.001.
        TOTAL_EPOCHES (int, optional) : Number of epochs to train the model before force end. Defaults to 1000.
        THRESH (float, optional) : Threshold for delta loss used in deciding early stop. Defaults to -infinity (off)
        verbose (bool, optional) : Turns on verbose output mode when set to True. Defaults to False.
        device (torch.device) : Device used to evaluate the model.

    Returns:
        Tuple[torch.nn.Module, list[Any], tuple[int, int]:
        Trained model, average loss per epoch during training,
        average loss per epoch during validation, and accuracy of validation per epoch.
    """
    # Global Vars.
    criterion = loss_function.to(device)
    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Running statistics
    training_tracker = []
    validation_tracker = []
    prev_loss = 100.
    accuracy = []

    for epoch in range(TOTAL_EPOCHES):
        if verbose and epoch % 10 == 0:                    # Verbose Logging
            print(f"\t\tEpoch {epoch+1}/{TOTAL_EPOCHES}")

        # Training
        model.train()

        training_loss_per_epoch: list[float] = []

        for inputs, class_targets in training_data:

            inputs = inputs.to(device)
            class_targets = class_targets.to(device)

            # Reset Gradient / Forward Pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute Loss
            loss = criterion(outputs, class_targets)

            # Backward Pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            training_loss_per_epoch.append(loss.item())

        training_tracker.append(np.mean(training_loss_per_epoch))

        # Validation
        avg_loss = evaluate(model, validation_data, validation_tracker, accuracy, device=device)

        if verbose and epoch % 10 == 0:                    # Verbose Logging
            print(f"\t\t\ttraining avg_loss={training_tracker[-1]:.4f}")
            print(f"\t\t\tvalidation avg_loss={avg_loss:.4f}")
            print(f"\t\t\taccuracy={accuracy[-1]:.2f}%")

        if abs(prev_loss - avg_loss) < THRESH: #early-stopping condition
            break
        prev_loss = avg_loss
    return model, training_tracker, validation_tracker, accuracy



def evaluate(
        model: Module,
        validation_data: DataLoader,
        validation_tracker : list[Any] = None,
        accuracy: list[Any] = None,
        device: torch.device = torch.device('cpu')
) -> floating[Any]:
    """
    Evaluates CNN model on a given set of data to evaluate statistics of average loss and accuracy

    Args:
        model (torch.nn.Module) : The model to be evaluated
        validation_data (DataLoader) : Data used to evaluate the model
        validation_tracker (list[Any], optional) : Tracker used to evaluate the model. Done in place.
        If none provided, function creates its own list and discards it.
        accuracy (list[Any], optional) : Accuracy of the model. Done in place.
        If none provided, function creates its own list and discards it.
        device (torch.device) : Device used to evaluate the model.

    Return
        (floating[Any]): Average loss for batch
    """

    if validation_tracker is None:
        validation_tracker = list()
    if accuracy is None:
        accuracy = list()
    criterion = loss_function.to(device)

    model.eval()

    model = model.to(device)

    correct = 0
    total = 0
    validation_loss_per_epoch: list[float] = []

    with torch.no_grad():
        for inputs, class_targets in validation_data:
            inputs = inputs.to(device)
            class_targets = class_targets.to(device)

            # Forward Pass
            outputs = model(inputs)

            # Compute Loss
            loss = criterion(outputs, class_targets)
            validation_loss_per_epoch.append(loss.item())

            # Apply 1-hot max to classify
            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == class_targets).sum().item()
            total += class_targets.size(0)

    # In place operations
    accuracy.append(100. * correct / total)  # convert to percentage
    avg_loss = np.mean(validation_loss_per_epoch)
    validation_tracker.append(avg_loss)
    return avg_loss