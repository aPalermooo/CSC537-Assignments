##############################################################################
# Name:           unpacker.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Randomly generates data to be used to train and test Linear Regression Models
#                  Using set weights
# Date:           30 January 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 1
##############################################################################
import numpy as np


np.random.seed(1)

# true weights
weights = (4,-3,2.5,-1,0.5)

distribution = 0
std = 1

dimensions = 5

def generate_data (size : int, name : str ) -> None:
    """
    Generates data according to a normal gaussian distribution
    :param size: the amount of data to be generated
    :param name: the file name to be saved
    :return: None
    """
    # generate the random data
    noise_vector = np.random.normal(distribution, std, size)
    feature_matrix = np.random.normal(distribution, std, (size, dimensions))

    # set file path to save to
    path = "dataset/"
    file_type = ".csv"
    file = path + name + file_type
    with open(file, "w") as f:
        for sample_number in range(size):

            feature_vector = feature_matrix[sample_number]

            # calc target
            #     sum(Feature_Value*Weight) + noise
            target = [
                    sum(
                        [feature_vector[feature_index] * weights[feature_index] for feature_index in range(dimensions)] + [noise_vector[0]]
                    )
                ]



            sample = np.concatenate((feature_vector, target))               # assemble output
            output = ','.join([str(value) for value in sample])+'\n'

            f.write(output)                                                  # append to file

if __name__ == "__main__":
    # Generate random datasets of set sizes and save them
    generate_data (6000, "training_data")
    generate_data (2000, "testing_data")