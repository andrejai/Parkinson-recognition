import csv
import os
import numpy as np
from util import normalize_data, feature_selection, divide_data_set, create_nn, get_data

def read(file_name="\datasets\\1\\train_data.csv"):
    X = []
    Y = []
    with open(file_name, 'rt', encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            Y.append([float(_x) for _x in row[26:28]])
            del row[26:28]
            X.append([int(_y) if _y.isdigit() else float(_y) for _y in row])
    return X, Y

def test(x, y, net):
    #simulate network, netOutput is y values
    netOutput = net.sim(x)

    #count number of right guesses
    updrs_true = 0
    existence = 0
    for i in range(0, len(netOutput)):
        if netOutput[i][0] - 0.5 <= y[i][0] <= netOutput[i][0] + 0.5:
            updrs_true += 1
        if y[i][0] > 0 and netOutput[i][0] > 0:
            existence += 1
        elif y[i][0] <= 0 and netOutput[i][0] <= 0:
            existence += 1
    print("\t\tPercentage of correctly counted UPDRS using NN is: {0}%" \
          .format((updrs_true / len(x)) * 100))
    print("\t\tPercentage of correctly recognized Parkinson using NN is: {0}%" \
          .format((existence / len(x)) * 100))


def train():
    x_data, y_data = read(os.getcwd() + "\datasets\\1\\train_data.csv")
    x_train, x_validation, x_test, y_train, y_validation, y_test = get_data(x_data, y_data)

    net, res = create_nn(x_train, y_train, os.getcwd() + "\\neural_rprop_ds1.net", 7, 1000, 0.5)
    test(x_test, y_test, net)