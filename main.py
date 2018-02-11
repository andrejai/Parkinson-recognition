import DS2 as DS2
import DS1 as DS1
import DS3 as DS3
import os
import numpy as np
from util import create_nn, normalize_data, feature_selection, divide_data_set, create_neural_network, find_minmax

if __name__ == "__main__":

    # Step 1: read data from file

    """

    x_data, y_data = DS2.read(os.getcwd() + "\datasets\\2\parkinsons_updrs.data.csv")


    # Step 2: feature selection
    reduces = [y[1] for y in y_data]
    x_data = feature_selection(np.array(x_data), np.array(reduces)).tolist()
    # Step 3: normalization
    x_data = normalize_data(x_data)
    y_data = normalize_data(y_data)

    # Step 4: find min and max value for each attribute
    minmax = find_minmax(x_data)
    # Step 5: divide data
    x_train, x_validation, x_test = divide_data_set(x_data)
    y_train, y_validation, y_test = divide_data_set(y_data)

    n_x_train = x_train
    n_y_train = [[y[1]] for y in y_train]
    # Step 6: train network
    net, res = create_neural_network(n_x_train, n_y_train, minmax, os.getcwd() + "\\neural_ds2.net")

"""
    DS2.train() # load data and train ds1 network
#    DS3.train()

