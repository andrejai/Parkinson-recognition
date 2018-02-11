import csv

import os

import numpy as np

from util import feature_selection, normalize_data, find_minmax, divide_data_set, create_neural_network, \
    denormalize_data


def read(file_name="parkinsons_udprs.data.csv"):
    X = []
    Y = []
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            Y.append([float(_x) for _x in row[4:6]])
            del row[4:6]
            X.append([int(_y) if _y.isdigit() else float(_y) for _y in row])
    return X, Y


def train():
    x_data, y_data = read(os.getcwd() + "\datasets\\2\parkinsons_updrs.data.csv")

    # Step 2: feature selection
    reduces = [y[1] for y in y_data]
    x_data = feature_selection(np.array(x_data), np.array(reduces)).tolist()
    # Step 3: normalization
    x_data, m, s = normalize_data(x_data)
    y_data, means, sqrt = normalize_data(y_data)

    # Step 4: find min and max value for each attribute
    minmax = find_minmax(x_data)
    # Step 5: divide data
    x_train, x_validation, x_test = divide_data_set(x_data)
    y_train, y_validation, y_test = divide_data_set(y_data)

    n_x_train = x_train
    n_y_train = [[y[1]] for y in y_train]
    # Step 6: train network
    net, res = create_neural_network(n_x_train, n_y_train, minmax, os.getcwd() + "\\neural_ds2_err260-1.net")

    test(x_test, y_test, net, means, sqrt)


def test(x, y, net, means, sqrt):
    # simulate network, netOutput is y values
    netOutput = net.sim(x)

    # count number of right guesses
    updrs_true = 0

    y = denormalize_data(y, means, sqrt)
    netOutput = denormalize_data(netOutput.tolist(), means, sqrt)
    print(y)
    print(netOutput)
    for i in range(0, len(netOutput)):
        if netOutput[i][0] - 0.5 <= y[i][1] <= netOutput[i][0] + 0.5:
            updrs_true += 1

    print("\t\tPercentage of correctly counted UPDRS using NN is: {0}%" \
          .format((updrs_true / len(x)) * 100))
