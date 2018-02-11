import DS2 as DS2
import DS1 as DS1
import DS3 as DS3
import os
import numpy as np
from util import create_nn, normalize_data, feature_selection, divide_data_set

if __name__ == "__main__":
    """
    x_data, y_data = DS2.read(os.getcwd() + "\datasets\\2\parkinsons_updrs.data.csv")
    x_train, x_validation, x_test = divide_data_set(x_data)
    y_train, y_validation, y_test = divide_data_set(y_data)
    reduces = [y[0] for y in y_train]
    fs = feature_selection(np.array(x_train), np.array(reduces))
    n_x_train = normalize_data(fs.tolist())
    n_y_train = normalize_data(y_train)

    net, res = create_nn(n_x_train, n_y_train, os.getcwd() + "\\neural_ds2.net")
"""
    DS1.train() # load data and train ds1 network
    DS3.train()

