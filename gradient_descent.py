import csv
import numpy as np


def get_data(filename) -> np.ndarray:
    path = "dataset/"
    extension = ".csv"

    with open(path + filename + extension) as f:
        return np.array(list(csv.reader(f, delimiter=',')))

def gen_prediction():

#     TODO:
# def gen_loss():
#
# def gen_gradient():

def gradient_descent( X,
                      Y,
                      w_init,
                      lr = 0.05,
                      max_iter = 3000,
                      tol=0.000001,
                      shuffle= True,
):
    print("Gradient Descent")

if __name__ == "__main__":
    data = get_data("sm_data")
    samples = np.delete(data, 5, 1)
    target = data.transpose()[5]

    print("Samples: ", samples)
    print("Target: ", target)
