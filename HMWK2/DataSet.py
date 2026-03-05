##############################################################################
# Name:           generator.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Randomly generates data
# Date:           6 March 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 2
##############################################################################
from dataclasses import dataclass
import torch


@dataclass
class DataSet:
    """Holds data set that can be used for training or validation"""

    features : torch.Tensor
    classifier : torch.Tensor
    targets : torch.Tensor