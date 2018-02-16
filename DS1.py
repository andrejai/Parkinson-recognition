import csv
import os

import numpy as np

from util import create_nn, test, feature_selection, normalize_data, divide_data_set


def read(file_name="\datasets\\1\\train_data.csv"):
    X = []
    Y = []
    with open(file_name, 'rt', encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            Y.append([float(_x) for _x in row[28:29]])
            del row[27:29]
            X.append([int(_y) if _y.isdigit() else float(_y) for _y in row])
    for y in Y:
        if y[0] == 0:
            y.append(1)
        else:
            y.append(0)
    return X, Y


def get_data(x_data, y_data):
    reduces = [y[0] for y in y_data]
    fs = feature_selection(np.array(x_data), np.array(reduces))
    n_x_data, means_x, sqrt_x = normalize_data(fs.tolist())

    x_train, x_validation, x_test = divide_data_set(n_x_data)
    y_train, y_validation, y_test = divide_data_set(y_data)
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def train():
    x_data, y_data = read(os.getcwd() + "\datasets\\1\\train_data.csv")

    x_train, x_validation, x_test, y_train, y_validation, y_test = get_data(x_data, y_data)
    path = input("Input output file name for neural network(enter for end): ")
    while path != "":
        path = path + ".net"
        net, res = create_nn(x_train, y_train, os.getcwd() + "\\"+path, 6, 1000, 0.01)
        test(x_test, y_test, net)
        path = input("Input output file name for neural network(enter for end): ")