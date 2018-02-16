import csv

import os

import numpy as np
import math

from util import feature_selection, normalize_data, find_minmax, divide_data_set, create_neural_network, \
    denormalize_data, minmax_normalize, minmax_denormalize


def read(file_name="parkinsons_udprs.data.csv"):
    X = []
    Y = []
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            Y.append([float(_x) for _x in row[4:6]])
            del row[4:6]
            del row[0]
            del row[2]
            X.append([int(_y) if _y.isdigit() else float(_y) for _y in row])
    return X, Y


def train():
    x_data, y_data = read(os.getcwd() + "\datasets\\2\parkinsons_updrs.data-sorted.csv")

    # Step 2: feature selection
    reduces = [y[0] for y in y_data]
    x_data = feature_selection(np.array(x_data), np.array(reduces)).tolist()
    # Step 3: normalization
    x_data, m, s = normalize_data(x_data)
    #x_data, mean, sqrt = normalize_data(y_data)
    y_data, maximums = minmax_normalize(y_data)

    # Step 4: find min and max value for each attribute
    minmax = find_minmax(x_data)
    # Step 5: divide data
    x_train, x_validation, x_test = divide_data_set(x_data)
    y_train, y_validation, y_test = divide_data_set(y_data)

    n_x_train = x_train[0:1000]
    n_y_train = [[y[0]] for y in y_train[0:1000]]
    # Step 6: train network
    net, res = create_neural_network(n_x_train, n_y_train, minmax, os.getcwd() + "\\ds2-10i753h1o-800e-.net")
    #test(x_test, y_test, net, 0,means,sqrt)
    test(x_test, y_test, net, maximums)
    # test(x_test, y_test,net)


def test(x, y, net, maksimums, means=0, sqrt=0):
    # simulate network, netOutput is y values
    netOutput = net.sim(x)

    # netOutput = net.predict(x)
    # netOutput = [[z] for z in netOutput]

    # count number of right guesses
    updrs_true = 0
    err_sum = 0
    y = minmax_denormalize(y, maksimums)
    netOutput = minmax_denormalize(netOutput.tolist(), maksimums)

    # y = denormalize_data(y, means,sqrt)
    # netOutput = minmax_denormalize(netOutput.tolist(), means,sqrt)
    for i in range(0, len(netOutput)):
        err = math.sqrt((y[i][0] - netOutput[i][0]) ** 2)
        err_sum += err
        if err:
            updrs_true += 1

    print("\t\tAverage error is: {0}%" \
          .format((err_sum / len(netOutput))))
    print("\t\tTotal error is: {0}%" \
          .format(err_sum))
