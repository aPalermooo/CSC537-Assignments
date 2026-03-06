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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pandas import DataFrame
from seaborn import FacetGrid
from torch.nn import init

from HMWK2.MLP import MLP, train
from HMWK2.generator import gen_data


DBUG = True     # turns on more logging features

# CONSTANTS

training_data, testing_data = gen_data()

input_dim = 10
hidden_dim = 5
output_dim = 1
output_classes = 2

num_hidden_layers = 2

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


def gen_plot(df: DataFrame,
             independent_var : str,
             experiment_number: int, experiment_title: str,
             dependent_var: str = "Loss",
             ) -> FacetGrid:
    """
    generates a grid of plots for the experiment
    :param df: DataFrame containing data to be plotted. Data Frame should have a column containing Iterations, and 2 columns mapping Independent Variable -> Dependent Variable
    :param independent_var: the variable that is being changed for each execution of the gradient descent algorithm
    :param experiment_number: The experiment number
    :param experiment_title: The experiment title
    :param dependent_var: (DEFAULT: LOSS) the value to be tracked over iterations
    :return:
    """
    g = sns.relplot(
        data=df,
        kind='line', x='Epoch', y=dependent_var,
        hue=independent_var, col=independent_var, legend=False,
        col_wrap=2
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Experiment {experiment_number}: {experiment_title}")
    for ax in g.axes.flat:
        ax.title.set_text(ax.get_title().replace(f"{independent_var} = ", ""))
    return g

###################################

def init_weights(model, activation):
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


def exp_1():
    # Activation Functions
    print("Staring Experiment 1...")
    experiment_number = 1
    experiment_title = "Activation Function"
    dv = "Loss"

    print("\t initializing...")
    independent_variables = ('ReLU', "Sigmoid", "Tanh")
    results = []

    for iv in independent_variables:

        model = MLP(input_dim, num_hidden_layers, hidden_dim, iv, output_dim)
        model.apply(lambda m: init_weights(m, iv))
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        model, loss_data = train(model, training_data, optimizer, False, thresh=0.001, verbose=DBUG)
        results.append(loss_data)

    # print(results)
    df = pd.DataFrame(data=results).transpose()
    df.columns = independent_variables
    # print(df)

    df['Epoch'] = df.index + 1

    df = df.melt(id_vars='Epoch', var_name=experiment_title, value_name=dv)
    df = df.dropna()
    print(df)

    # g = gen_plot(df, experiment_title, experiment_number, dv)

    g = sns.relplot(data=df, x="Epoch", y="Loss",
                kind="line", hue="Activation Function", hue_order=independent_variables)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Experiment {experiment_number}: {experiment_title}")

    save_plot(g, experiment_number, experiment_title)




def exp_2():
    print("hello world")

def exp_3():
    print("hello world")

def exp_4():
    print("hello world")

def main():
    exp_1()

if __name__ == "__main__":
    main()