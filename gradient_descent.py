##############################################################################
# Name:           gradient_descent.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Implements the Linear Regression model discussed during class
# Date:           30 January 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 1
##############################################################################
import csv
import numpy as np

from HMWK1.logger import Logger


# ----------------------
# HELPER FUNCTIONS

def get_data(filename : str ) -> (np.ndarray, np.ndarray):
    """
    Locates a file containing samples and targets and compiles them into a numpy matrix
    :param filename: The name of the file; must be in dataset directory and be a csv file
    :return: 2 numpy matrices; one containing all feature samples and the other containing their respective targets
    """
    path = "dataset/"
    extension = ".csv"

    with open(path + filename + extension) as f:
        data = np.array(list(csv.reader(f, delimiter=',')), dtype = np.float64)
        return np.delete(data, 5, 1), data.transpose()[5]

def shuffle_data(set1, set2):
    """
    shuffles two numpy matrices and maintains indices between them
    :raises ValueError if matrices do not contain the same number of rows
    :param set1: matrix 1
    :param set2: matrix 2
    :return: 2 matrices in shuffled order
    """
    assert len(set1) == len(set2)
    order = np.random.permutation(len(set1))
    return set1[order], set2[order]

def gen_prediction(weights, features):
    """
    Estimates the value of the target using the value of the current calculated weights and the sample inputs
    :param weights: the current calculated weights
    :param features: the values of the feature samples
    :return: a predicted value of what the target will be
    """
    return np.dot(features, weights)  # Xw

def gen_mse(target, prediction):
    """
    Calculates the mean squared error
    :param target: the target value (actual value the samples generate)
    :param prediction: the predicted value (the value the samples were estimated to be)
    :return: a value that represents the cost of the prediction
    """
    mse = np.sum(np.square(target - prediction))  # MSE calc
    mse = 1 / (2 * target.shape[0]) * mse
    return mse

def gen_l2 (weights, hyperparam):
    """
    Calculates weight decay (moves weight closer to 0) across ALL weights
    :param weights: the current calculated value of the weights
    :param hyperparam: the intensity of weight decay
    :return: a value of total weight decay
    """
    return hyperparam * np.sum(np.square(weights)) # lambda = 0 -> skipped

def gen_gradient(weights, features, target, hyperparam=0.0):
    """
    applies gradient descent with optional weight decay
    :param weights: the current calculated value of the weights
    :param features: the values of the feature samples
    :param target: the target value (actual value the samples generate)
    :param hyperparam: (optional) the intensity of weight decay
    :return: vector containing new calculated weights in respect to cost function
    """
    batch_size = target.shape[0]
    prediction = gen_prediction(weights, features)

    error = prediction - target  # Xw-y

    return ((1 / batch_size)
            * (np.dot(features.transpose(), error))
            + (2 * hyperparam * weights))

# ----------------------

def gradient_descent( X : np.ndarray, Y : np.ndarray, w_init : np.ndarray,
                      lr : float = 0.05,
                      batch_size : int = 64,
                      weight_decay :float = 0.0,
                      max_iter : int = 3000,
                      tol : float = 0.000001,
                      shuffle : bool = True,
                    ) -> Logger:
    """
    implements the gradient descent algorithm and tracks statistics over iterations
    :param X: The training sample data
    :param Y: The training target data
    :param w_init: The initial value of each weight (5)
    :param lr: The learning rate; how much weights are adjusted in each iteration
    :param batch_size: How many samples are examined in an iteration
    :param weight_decay: Amount of decay applied to each weight (bringing them to 0)
    :param max_iter: Manual stop (how many batches to process)
    :param tol: How much difference in cost function from past epoch is accepted as being convergent
    :param shuffle: Shuffle the data between every epoch
    :return: a logger class containing various statistics of run-time
    """
    # init counting variables
    iterations = 0
    epoches = 0

    prev_loss = 0.0
    max_loss = -np.inf

    w = w_init

    output = Logger()   # see HWMK1/logger.py

    while True:                                                   # Until a stopping criteria is met
        #start epoch

        if shuffle:                                               # shuffle for every epoch (if enabled)
            X, Y = shuffle_data(X, Y)

        for batch_i in range(0, X.shape[0], batch_size):
            #start iteration

            batch_end = batch_i + batch_size

            ep_x = X[batch_i:batch_end]                            # create batch samples and targets
            ep_y = Y[batch_i:batch_end]

            w -= lr * gen_gradient(w, ep_x, ep_y, weight_decay)    # adjust weights

            prediction = gen_prediction(w, ep_x)                    # log loss

            loss = gen_mse(ep_y, prediction) + gen_l2(w, weight_decay)

            output.weights_data.append(w.copy())
            output.loss_data.append(loss)
            output.mse_data.append(gen_mse(ep_y, prediction))
            output.l2_data.append(np.sum(np.square(w)))

            iterations += 1

            if iterations >= max_iter:  # User defined stop
                return output
            # print(iterations, abs(loss - prev_loss))
            max_loss = max(max_loss, loss)

            #end iteration

        if abs(max_loss - prev_loss) < tol:  # Stop on convergence
            return output

        # prepare for next epoch
        prev_loss = max_loss
        epoches += 1

        #end epoch

def test_weights(X, Y, log, hyperparam = 0.0):
    """
    Runs gradient descent algorithm, but edits parameters such that weights are not adjusted
    :param X: The testing sample data
    :param Y: The testing target data
    :param log: the weights used in the training set
    :param hyperparam: (optional) the intensity of weight decay
    :return: a logger class containing various statistics of run-time
    """
    output = Logger()
    for w in log.weights_data:
        prediction = gen_prediction(w, X)

        mse = gen_mse(Y, prediction)
        l2 = gen_l2(w, hyperparam)
        loss = mse + l2

        output.loss_data.append(loss)
        output.mse_data.append(mse)
        output.l2_data.append(l2)
    return output

def main() -> None:
    """
    Runs a test of gradient descent and testing function
    for dbug
    :return: None
    """
    # Var init
    training_file = "training_data"
    test_file = "testing_data"

    start_weights = [0, 0, 0, 0, 0]

    # Experiment preparation
    training_sample, training_target = get_data(training_file)
    test_sample, test_target = get_data(test_file)

    start_weights = np.array(start_weights, dtype=np.float64)

    # Experiment
      # Training
    runtime_data = gradient_descent(training_sample, training_target, start_weights)

      # Testing
    test_results = test_weights(test_sample, test_target, runtime_data)

    # print(len(runtime_data.weights_data))

    # plt.plot(test_results.mse_data)
    # plt.show()
    print(runtime_data.weights_data[-1])

if __name__ == "__main__":
    main()