##############################################################################
# Name:           logger.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Helper class used to log data over iteration throughout
#                   the execution of Gradient Descent Functino
# Date:           30 January 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 1
##############################################################################
from dataclasses import dataclass

import numpy as np


@dataclass
class Logger:
    """Logs various statistics about the linear regression algorithm."""
    weights_data : list[np.ndarray]
    loss_data : list[float]
    mse_data: list[float]
    l2_data: list[float]

    def __init__(self):
        self.weights_data = []
        self.loss_data = []
        self.mse_data = []
        self.l2_data = []