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
from torch.nn import Module
from torch.utils.data import DataLoader


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
        """
        layers = []

        for _ in range(depth):
            layers.append(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            )
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU)
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
        data : DataLoader,
        optimizer : torch.optim.Optimizer,
        TOTAL_EPOCHES : int = 1000,
        THRESH : float = float('-inf'),
        verbose : bool = False,
        ) -> tuple[Module, list[Any]]:
    """
    Trains CNN model on a given set of data

    Args:
        model (torch.nn.Module) : The model to be trained
        data (DataLoader)       : Data used to train the model
        optimizer (torch.optim.Optimizer) : Optimizer used to train the model
        TOTAL_EPOCHES (int, optional) : Number of epochs to train the model before force end. Defaults to 1000.
        THRESH (float, optional) : Threshold for delta loss used in deciding early stop. Defaults to -infinity (off)
        verbose (bool, optional) : Turns on verbose output mode when set to True. Defaults to False.

    Returns:
        Tuple[torch.nn.Module, list[Any]]: Trained model and average loss per epoch
    """
    # Global Vars.
    criterion = torch.nn.CrossEntropyLoss()

    # Running statistics
    loss_tracker = []
    prev_loss = 100.

    for epoch in range(TOTAL_EPOCHES):
        if verbose and epoch % 100 == 0:                    # Verbose Logging
            print(f"\t\tEpoch {epoch+1}/{TOTAL_EPOCHES}")

        model.train()

        loss_per_epoch: list[float] = []

        for inputs, class_targets in data:

            # Reset Gradient / Forward Pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute Loss
            loss = criterion(outputs, class_targets)

            # Backward Pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            loss_per_epoch.append(loss.item())

        avg_loss = np.mean(loss_per_epoch)
        loss_tracker.append(avg_loss)

        if verbose and epoch % 100 == 0:                    # Verbose Logging
            print(f"\t\t\t{avg_loss=}")

        if abs(prev_loss - avg_loss) < THRESH: #stopping condition
            break
        prev_loss = avg_loss
    return model, loss_tracker


