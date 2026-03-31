##############################################################################
# Name:           plotter.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Conducts the specified experiments on the configurable MLP neural network
# Date:           6 March 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 2
##############################################################################
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pandas import DataFrame
from seaborn import FacetGrid
from torch.distributions.constraints import independent
from torch.nn import init, Module

from HMWK2.MLP import MLP, train, evaluate
from HMWK2.generator import gen_data


DBUG = False     # turns on more logging features

# CONSTANTS
## Remain static through experiments (if they are not the independent variable)
training_data, testing_data = gen_data()

input_dim = 10
hidden_dim = 32
output_dim = 1
output_classes = 2

num_hidden_layers = 2
act_fn = 'ReLU'
learning_rate = 2e-3
threshold = 0.001


plot_path = "plots/"
file_type = ".png"

############################

# Helper Functions

## Imports from Assignment 1

def save_plot (exp_plot, exp_num, exp_name, suffix = "") -> None:
    """
    Saves a plot to designated location
    :param exp_plot: the experiment's plot
    :param exp_num: the experiment's number
    :param exp_name: the experiment's name
    :param suffix: (optional) a title suffix to differentiate plots from the same experiment
    :return: None
    """
    complete_path = plot_path + f"exp_{exp_num}/{exp_name}" + suffix + file_type
    exp_plot.savefig(complete_path)
    print(f'\t\t saved to {complete_path}')

###################################

def init_weights(model, activation) -> None:
    """
    Generate random weights using methodology depending on what works best with a model's activation function
    :param model: the model that is being configured
    :param activation: the activation function the model uses
    :return: None; done in place
    """
    if not isinstance(model, torch.nn.Linear):
        return

    match activation.lower():
        case "relu":
            init.kaiming_uniform_(model.weight, mode='fan_in', nonlinearity='relu')
        case "sigmoid" | "tanh":
            init.xavier_normal_(model.weight)
        case _:
            raise ValueError("Unknown activation function")
    init.zeros_(model.bias)

def eval_regression(iv: str, loss_data: list[float], model: Module) -> None:
    """
    Calculates accuracy statistics of a model based on how it performs to test data.
    Also logs important information about the trial to console
    :param iv: the value ot the variable being tested [for logging purposes]
    :param loss_data: performance on training data [for logging purposes]
    :param model: the model to be evaluated
    :return: None; just logs data
    """
    loss_values = evaluate(model, testing_data, is_classification=False)
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)
    print(f"\t\t{iv:30}- Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"\t\t\t\tCompleted in {len(loss_data)} epochs.")

def gen_plot(dv: str, experiment_number: int, experiment_title: str, independent_variables,
             results: list[Any]) -> FacetGrid:
    """
    Generates a line plot to compare trials of the experiment to analyze how the independent variable effected outcomes on
    convergence and accuracy. Also applies a title and legend to each figure.
    :param dv: the description of what is being measured to evaluate results.
    :param experiment_number: the id of the experiment being conducted.
    :param experiment_title: the title of the experiment being conducted.
    :param independent_variables: a list of the value that the independent variable was tried with for each trial.
    :param results: the results yield by each trial. A list of trials, each containing a list of linear data points.
    :return: the figure generated.
    """
    ## Create DataFrame
    df = pd.DataFrame(data=results).transpose()
    df.columns = independent_variables

    df['Epoch'] = df.index + 1

    df = df.melt(id_vars='Epoch', var_name=experiment_title, value_name=dv)
    df = df.dropna()

    ## Plot results & save
    g = sns.relplot(data=df, x="Epoch", y=dv,
                    kind="line", hue=experiment_title, hue_order=independent_variables)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Experiment {experiment_number}: {experiment_title}")
    return g

#################################################

# EXPERIMENTS

def exp_1() -> None:
    """
    Tests how different activation functions effect convergence speed and model accuracy
    :return: Saves plot to experiment directory
    """
    # Set up experiment
    experiment_number = 1
    experiment_title = "Activation Function"
    dv = "Loss"

    print(f"Staring Experiment {experiment_number}...")


    print("\t initializing...")
    independent_variables = ('ReLU', "Sigmoid", "Tanh")
    results = []

    print("\tsimulating models...")
    # Architecture 1: Regression
    print("\t simulating regression models....")
    for iv in independent_variables:

        # Create network
        model = MLP(input_dim, num_hidden_layers, hidden_dim, iv, output_dim)
        model.apply(lambda m: init_weights(m, iv))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        model, loss_data = train(model, training_data, optimizer, False, thresh=threshold, verbose=DBUG)
        results.append(loss_data)

        # Test
        eval_regression(iv, loss_data, model)

    print("\t  plotting results...")
    g = gen_plot(dv, experiment_number, experiment_title, independent_variables, results)

    save_plot(g, experiment_number, experiment_title, suffix=" - Regression")

    # Architecture 2: Classification
    results = []

    print("\t simulating classification models....")
    for iv in independent_variables:
        # Create Network
        model = MLP(input_dim, num_hidden_layers, hidden_dim, iv, output_classes)
        model.apply(lambda m: init_weights(m, iv))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        model, loss_data = train(model, training_data, optimizer, True, thresh=threshold, verbose=DBUG)
        results.append(loss_data)

        # Test
        loss_values, accuracy = evaluate(model, testing_data, is_classification=True)
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        print(f"\t\t{iv:30}- Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"\t\t\t\tAccuracy: {accuracy * 100:.2f}%")
        print(f"\t\t\t\tCompleted in {len(loss_data)} epochs.")

    print("\t  plotting results...")
    g = gen_plot(dv, experiment_number, experiment_title, independent_variables, results)

    # plt.show()
    save_plot(g, experiment_number, experiment_title, " - Classification")

def exp_2() -> None:
    """
    Tests how different compositions of hidden layers effect the convergence speed and accuracy
    :return: Saves plot to experiment directory
    """
    #Depth vs. Width
    # Set up experiment
    experiment_number = 2
    experiment_title = "Depth vs. Width"
    dv = "Loss"

    print(f"Staring Experiment {experiment_number}...")


    print("\t initializing...")
    iv_depths = (1, 3)
    iv_widths = (50, 20)
    results = []

    print("\tsimulating models...")
    # Architecture 1: Regression
    print("\t simulating regression models....")
    for depth, width in zip(iv_depths, iv_widths):
        # Create Network
        model = MLP(input_dim, depth, width, act_fn, output_dim)
        model.apply(lambda m: init_weights(m, act_fn))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        model, loss_data = train(model, training_data, optimizer, False, thresh=threshold, verbose=DBUG)
        results.append(loss_data)

        # Evaluate
        eval_regression(f"{depth}x{width}", loss_data, model)


    print("\t  plotting results...")
    col_names = [f"{d}x{w}" for d, w in zip(iv_depths, iv_widths)]
    g = gen_plot(dv, experiment_number, experiment_title, col_names, results)

    # plt.show()
    save_plot(g, experiment_number, experiment_title, " - Regression")


    # Architecture 2: Classification
    results = []

    print("\t simulating classification models....")
    for depth, width in zip(iv_depths, iv_widths):
        # Create Network
        model = MLP(input_dim, depth, width, act_fn, output_classes)
        model.apply(lambda m: init_weights(m, act_fn))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        model, loss_data = train(model, training_data, optimizer, True, thresh=threshold, verbose=DBUG)
        results.append(loss_data)

        # Test
        loss_values, accuracy = evaluate(model, testing_data, is_classification=True)
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        print(f"\t\t{f"{depth}x{width}":30}- Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"\t\t\t\tAccuracy: {accuracy*100:.2f}%")
        print(f"\t\t\t\tCompleted in {len(loss_data)} epochs.")

    print("\t  plotting results...")
    g = gen_plot(dv, experiment_number, experiment_title, col_names, results)

    # plt.show()
    save_plot(g, experiment_number, experiment_title, " - Classification")

def exp_3() -> None:
    """
    Tests how different optimization functions effect the convergence speed and accuracy of an MLP network
    :return: Saves plot to experiment directory
    """
    # Set up experiment
    experiment_number = 3
    experiment_title = "Optimizers"
    dv = "Loss"

    print(f"Staring Experiment {experiment_number}...")

    print("\t initializing...")
    independent_variables = ("SGP (Momentum)", "ADAM")  #Since there are only 2 iv's, their implementations are more manual
                                                            # This saves complexity of coming up with a way to dynamically generate a different optimizer
    results = []

    print("\tsimulating models...")
    # Architecture 1: Regression
    print("\t simulating regression models....")
    for index, iv in enumerate(independent_variables):
        # Create Network
        model = MLP(input_dim, num_hidden_layers, hidden_dim, act_fn, output_dim)
        model.apply(lambda m: init_weights(m, act_fn))
        if index == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        # Train
        model, loss_data = train(model, training_data, optimizer, False, thresh=threshold, verbose=DBUG)
        results.append(loss_data)

        # Test
        eval_regression(iv, loss_data, model)

    print("\t  plotting results...")
    g = gen_plot(dv, experiment_number, experiment_title, independent_variables, results)

    # plt.show()
    save_plot(g, experiment_number, experiment_title, " - Regression")

    # Architecture 2: Classification
    results = []

    print("\t simulating classification models....")
    for index, iv in enumerate(independent_variables):
        # Create Network
        model = MLP(input_dim, num_hidden_layers, hidden_dim, act_fn, output_classes)
        model.apply(lambda m: init_weights(m, act_fn))
        if index == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        model, loss_data = train(model, training_data, optimizer, True, thresh=threshold, verbose=DBUG)
        results.append(loss_data)

        # Test
        loss_values, accuracy = evaluate(model, testing_data, is_classification=True)
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        print(f"\t\t{iv:30}- Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"\t\t\t\tAccuracy: {accuracy * 100:.2f}%")
        print(f"\t\t\t\tCompleted in {len(loss_data)} epochs.")

    print("\t  plotting results...")
    g = gen_plot(dv, experiment_number, experiment_title, independent_variables, results)

    # plt.show()
    save_plot(g, experiment_number, experiment_title, " - Classification")

def exp_4() -> None:
    """
    Tests how L2 weight decay effects the convergence speed and accuracy of an MLP network
    :return: Saves plot to experiment directory
    """
    experiment_number = 4
    experiment_title = "Regularization"
    dv = "Loss"

    print(f"Staring Experiment {experiment_number}...")

    print("\t initializing...")
    independent_variables = (0., 0.001)
    results = []

    print("\tsimulating models...")
    # Architecture 1: Regression
    print("\t simulating regression models....")
    for iv in independent_variables:
        # Create Network
        model = MLP(input_dim, num_hidden_layers, hidden_dim, act_fn, output_dim,)
        model.apply(lambda m: init_weights(m, act_fn))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        # Train
        model, loss_data = train(model, training_data, optimizer, False, thresh=threshold, verbose=DBUG, l2=iv)
        results.append(loss_data)

        # Test
        eval_regression(str(iv), loss_data, model)

    print("\t  plotting results...")
    g = gen_plot(dv, experiment_number, experiment_title, independent_variables, results)

    # plt.show()
    save_plot(g, experiment_number, experiment_title, " - Regression")

    # Architecture 2: Classification
    results = []

    print("\t simulating classification models....")
    for iv in independent_variables:
        # Create Network
        model = MLP(input_dim, num_hidden_layers, hidden_dim, act_fn, output_classes)
        model.apply(lambda m: init_weights(m, act_fn))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        model, loss_data = train(model, training_data, optimizer, True, thresh=threshold, verbose=DBUG, l2=iv)
        results.append(loss_data)

        # Test
        loss_values, accuracy = evaluate(model, testing_data, is_classification=True)
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        print(f"\t\t{iv:30}- Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"\t\t\t\tAccuracy: {accuracy * 100:.2f}%")
        print(f"\t\t\t\tCompleted in {len(loss_data)} epochs.")

    print("\t  plotting results...")
    g = gen_plot(dv, experiment_number, experiment_title, independent_variables, results)

    # plt.show()
    save_plot(g, experiment_number, experiment_title, " - Classification")

####################################################

def main() -> None:
    """
    Runs all experiments and applies unifying theme to plots
    :return: None
    """
    sns.set_style("whitegrid")
    sns.set_palette("bright")

    exp_1()
    exp_2()
    exp_3()
    exp_4()
    # plt.show()

if __name__ == "__main__":
    main()