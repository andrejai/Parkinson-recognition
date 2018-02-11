import csv
import os
import numpy as np
from util import create_nn, get_data, denormalize_data

def read(file_name="\datasets\\1\\train_data.csv"):
    X = []
    Y = []
    with open(file_name, 'rt', encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            Y.append([float(_x) for _x in row[27:29]])
            del row[27:29]
            X.append([int(_y) if _y.isdigit() else float(_y) for _y in row])
    return X, Y

def test(x, y, net, means, sqrt):
    #simulate network, netOutput is y values
    netOutput = net.sim(x)

    #count number of right guesses
    updrs_true = 0
    existence = 0
    y = denormalize_data(y, means, sqrt)
    netOutput = denormalize_data(netOutput.tolist(), means, sqrt)
    print(y)
    print(netOutput)
    for i in range(0, len(netOutput)):
        if netOutput[i][0] - 0.5 <= y[i][0] <= netOutput[i][0] + 0.5:
            updrs_true += 1
        if y[i][0] > 0.5 and netOutput[i][0] > 0.5:
            existence += 1
        elif y[i][0] <= 0.5 and netOutput[i][0] <= 0.5:
            existence += 1
    print("\t\tPercentage of correctly counted UPDRS using NN is: {0}%" \
          .format((updrs_true / len(x)) * 100))
    print("\t\tPercentage of correctly recognized Parkinson using NN is: {0}%" \
          .format((existence / len(x)) * 100))


def train():
    x_data, y_data = read(os.getcwd() + "\datasets\\1\\train_data.csv")
    print(y_data)
    x_train, x_validation, x_test, y_train, y_validation, y_test, means, sqrt = get_data(x_data, y_data)
    print(y_train)
    net, res = create_nn(x_train, y_train, os.getcwd() + "\\ds1_7_1000_01_rprop.net", 7, 1000, 0.01)
    test(x_test, y_test, net, means, sqrt)

