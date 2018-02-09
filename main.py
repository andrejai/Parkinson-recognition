import DS2 as DS2
import os
from util import create_nn, normalize_data

if __name__ == "__main__":
    print(os.getcwd())
    x_data, y_data = DS2.read(os.getcwd() + "\datasets\\2\parkinsons_updrs.data.csv")
    x_train, x_validation, x_test = DS2.divide_data_set(x_data)
    y_train, y_validation, y_test = DS2.divide_data_set(y_data)
    n_x_train = normalize_data(x_train)
    n_y_train = normalize_data(y_train)

    net, res = create_nn(n_x_train, n_y_train, os.getcwd() + "\\neural_ds2.net")
