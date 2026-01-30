##############################################################################
# Name:           plotter.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Runs experiments as described in the assignment and generates/saves plots
#                     pertaining to the requested information
# Date:           30 January 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 1
##############################################################################

import numpy as np
import seaborn as sns
import pandas as pd
from pandas import DataFrame
from seaborn import FacetGrid
from HMWK1.gradient_descent import get_data, gradient_descent, test_weights

# constants
training_file = "training_data"
test_file = "testing_data"

plot_path = "experiments/"
file_type = ".png"

training_sample, training_target = get_data(training_file)
test_sample, test_target = get_data(test_file)

num_features = len(training_sample[0])

# ----------------------
# HELPER FUNCTIONS

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
        kind='line', x='Iteration', y=dependent_var,
        hue=independent_var, col=independent_var, legend=False,
        col_wrap=2
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Experiment {experiment_number}: {experiment_title}")
    for ax in g.axes.flat:
        ax.title.set_text(ax.get_title().replace(f"{independent_var} = ", ""))
    return g

# ----------------------


def exp_1 () -> None:
    """
    Tests how different weights effects gradient descent
    :return: Saves plot to experiment directory
    """
    experiment_number = 1
    experiment_title = "Weight Distribution"

    print("Staring Experiment 1...")

    "\t initializing..."
    # const
    learn_rate = 0.05
    batch_size = 64
    hyper_param = 0

    # independent variable
    zero_weights = np.array([0]*num_features, dtype=float)
    gaussian_weights = np.array(np.random.normal(loc=0.0, scale=0.1, size=num_features), dtype=float)
    uniform_weights = np.array(np.random.uniform(low=-.1, high=.1, size=num_features), dtype=float)

    print("\t generating models...")    #using test data

    zero_results = gradient_descent(training_sample, training_target, zero_weights, learn_rate, batch_size, hyper_param)
    gaussian_results = gradient_descent(training_sample, training_target, gaussian_weights, learn_rate, batch_size, hyper_param)
    uniform_results = gradient_descent(training_sample, training_target, uniform_weights, learn_rate, batch_size, hyper_param)

    print("\t testing models...")

    zero_test = test_weights(test_sample, test_target, zero_results, hyper_param)
    gaussian_test = test_weights(test_sample, test_target, gaussian_results, hyper_param)
    uniform_test = test_weights(test_sample, test_target, uniform_results, hyper_param)

    print("\t plotting results...")

    df = pd.DataFrame({
        'Zeroes Training': zero_results.mse_data,
        'Gaussian Training': gaussian_results.mse_data,
        'Uniform Training': uniform_results.mse_data,
        'Zeroes Test': zero_test.mse_data,
        'Gaussian Test': gaussian_test.mse_data,
        'Uniform Test': uniform_test.mse_data,
    })

    # Clean data for plotting
    df['Iteration'] = df.index
    df = pd.melt(df, id_vars='Iteration', value_name='MSE')             #Transform to long form data

    df['Weights Variable'] = df['variable'].str.replace(r' (Training|Test)$', '', regex=True)   # Splice variable column to separate
    df['Data Set'] = df['variable'].str.extract(r'(Training|Test)$')                                # Training and Testing data
    df = df.drop(['variable'], axis=1)


    # Plotting data (Requires a specific plot not compatible with helper function)
    g = sns.relplot(
        data=df,
        kind='line', x='Iteration', y='MSE',hue='Weights Variable', col='Data Set'
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Experiment {experiment_number}: {experiment_title}")
    g.axes.flat[0].title.set_text('Training Set')
    g.axes.flat[1].title.set_text('Testing Set')

    save_plot(g,experiment_number,experiment_title)


def exp_2 () -> None:
    """
    Tests how different learning weights effect the gradient descent algorithm
    :return: Saves plot to experiment directory
    """
    experiment_number = 2
    experiment_title = "Learning Rate Selection"
    print("Staring Experiment 2...")
    print("\t initializing...")

    # const
    batch_size = 64
    hyper_param = 0
    weights = np.array([0]*num_features, dtype=float)

    # independent variable
    iv = "Learning Rate"
    learn_rates = [0.01, 0.5, 0.1, 0.2]

    df = pd.DataFrame(columns=['Iteration', iv, 'Loss'])

    print("\t generating models...")
    for learn_rate in learn_rates:
        results = gradient_descent(training_sample, training_target, weights, learn_rate, batch_size, hyper_param)
        results = results.loss_data
        result_df = pd.DataFrame({
            'Iteration': range(len(results)),
            iv: [learn_rate] * len(results),
            'Loss': results
        })

        df = pd.concat([df, result_df], ignore_index=True) # build DataFrame as long-form data

    print("\t plotting results...")

    # Ensure DataFrame typing
    df['Iteration'] = df['Iteration'].astype(int)
    df[iv] = df[iv].astype(str)
    df['Loss'] = df['Loss'].astype(float)

    g = gen_plot(df, iv, experiment_number, experiment_title)

    save_plot(g,experiment_number,experiment_title)

def exp_3 () -> None:
    """
    Tests how different Batch Sizes effect the gradient descent algorithm
    :return: Saves plot to experiment directory
    """
    experiment_number = 3
    experiment_title = "Batch Size Comparison"

    print("Staring Experiment 3...")
    print("\t initializing...")

    # const
    lr = 0.05
    hyper_param = 0
    weights = np.array([0]*num_features, dtype=float)

    # independent variable
    iv = "Batch Size"
    batch_sizes = [1,16,64,256,1024]

    print("\t generating models...")

    df = pd.DataFrame(columns=['Iteration', iv, 'Loss'])

    for batch_size in batch_sizes:
        results = gradient_descent(training_sample, training_target, weights, lr, batch_size, hyper_param)
        results = results.loss_data
        result_df = pd.DataFrame({
            'Iteration': range(len(results)),
            iv: [batch_size] * len(results),
            'Loss': results
        })

        df = pd.concat([df, result_df], ignore_index=True)  # build DataFrame as long-form data

    print("\t plotting results...")


    # Ensure DataFrame typing
    df['Iteration'] = df['Iteration'].astype(int)
    df[iv] = df[iv].astype(str)
    df['Loss'] = df['Loss'].astype(float)

    g = gen_plot(df, iv, experiment_number, experiment_title)

    save_plot(g,experiment_number,experiment_title)

def exp_4 () -> None:
    """
    Tests how different amounts of Weight Decay effect the gradient descent algorithm
    :return: Saves plots to experiment directory
    """
    experiment_number = 4
    experiment_title = "Effect of Weight Decay Regularization"

    print("Staring Experiment 4...")
    print("\t initializing...")

    # const
    lr = 0.05
    batch_size = 64
    weights = np.array([0]*num_features, dtype=float)

    # independent variable
    iv = "Weight Decay"
    hyper_params = [0, 0.001 ]

    print("\t generating models...")

    df = pd.DataFrame(columns=['Iteration', iv, 'MSE', 'L2', 'Data Set'])

    for hyper_param in hyper_params:

        # Training algorithm

        training_results = gradient_descent(training_sample, training_target, weights, lr, batch_size, hyper_param)
        mse_data = training_results.mse_data
        l2_data = training_results.l2_data
        results_df = pd.DataFrame({
            'Iteration': range(len(mse_data)),
            iv: [hyper_param] * len(mse_data),
            'MSE': mse_data,
            'L2': l2_data,
            'Data Set': ["Training"] * len(mse_data)
        })

        df = pd.concat([df, results_df], ignore_index=True)   # build DataFrame as long-form data


        # Testing Algorithm
        test_results = test_weights(test_sample, test_target, training_results, hyper_param)
        mse_data = test_results.mse_data
        l2_data = test_results.l2_data
        results_df = pd.DataFrame({
            'Iteration': range(len(mse_data)),
            iv: [hyper_param] * len(mse_data),
            'MSE': mse_data,
            'L2': l2_data,
            'Data Set': ["Test"] * len(mse_data)
        })

        df = pd.concat([df, results_df], ignore_index=True) # build DataFrame as long-form data

    print("\t testing models...")

    # Ensure DataFrame typing
    print("\t plotting results...")
    df['Iteration'] = df['Iteration'].astype(int)
    df[iv] = df[iv].astype(str)
    df['MSE'] = df['MSE'].astype(float)
    df['L2'] = df['L2'].astype(float)
    df['Data Set'] = df['Data Set'].astype(str)

    # Plotting data (Requires a specific plot not compatible with helper function)
    for dv in ['MSE', 'L2']:        # Generates 2 plots
        g = sns.relplot(
            data=df,
            kind='line', x='Iteration', y=dv, hue=iv, col='Data Set'
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Experiment {experiment_number}:{experiment_title}")
        g.axes.flat[0].title.set_text('Training Set')
        g.axes.flat[1].title.set_text('Testing Set')

        save_plot(g,experiment_number,experiment_title, f' - {dv}')


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

if __name__ == "__main__":
    main()