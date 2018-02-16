import os
import neurolab as nl
import numpy as np
import math
from sklearn.feature_selection import SelectKBest, f_regression


def feature_selection(data, target):
    new_input = SelectKBest(f_regression, k=10)
    new_input.fit(data, target)
    result = new_input.transform(data)

    return result


def normalize_data(input_data):
    # http://www.d.umn.edu/~deoka001/Normalization.html

    # calculate mean for each columng
    means = np.mean(input_data, axis=0)
    # calculate (x - mean)^2
    subtracted = [[(x - means[y.index(x)]) ** 2 for x in y] for y in input_data]
    # deviation is the sqrt of mean
    deviation = np.mean(subtracted, axis=0)
    sqrt = [math.sqrt(x) for x in deviation]
    # normalize the values (x-mean)/deviation
    normalized = [[(x - means[y.index(x)]) / sqrt[y.index(x)] for x in y] for y in input_data]

    return normalized, means, sqrt


def minmax_normalize(input_data):
    maximums = np.max(input_data, axis=0)
    normalized = [[x / maximums[y.index(x)] for x in y] for y in input_data]
    return normalized, maximums


def minmax_denormalize(input_data, maximums):
    normalized = [[x * maximums[y.index(x)] for x in y] for y in input_data]
    return normalized


def find_minmax(input):
    mins = np.min(input, axis=0).tolist()
    maxs = np.max(input, axis=0).tolist()
    minmax = [[math.floor(x), math.ceil(maxs[mins.index(x)])] for x in mins]
    return minmax


def denormalize_data(input_data, means, sqrt):
    denormalized = [[x * sqrt[y.index(x)] + means[y.index(x)] for x in y] for y in input_data]
    return denormalized


def denormalize_data(input_data, means, sqrt):

    denormalized = [[x * sqrt[y.index(x)] + means[y.index(x)] for x in y] for y in input_data]
    return denormalized


def create_nn(input_data, output_data, path, hidden_layers=8, epochs=500, goal=0.5):

    if os.path.isfile(path):
        print("\n\tLoading network from " + path)
        net = nl.load(path)
        res = net.sim(input_data)
        return net, res

    # init the neural network
    net = nl.net.newff([[-100, 100]] * len(input_data[0]), [hidden_layers, len(output_data[0])])
    net.trainf = nl.net.train.train_rprop
    net.init()
    # train the neural network
    net.train(input_data, output_data, epochs=epochs, show=10, goal=goal)

    # save network to a file
    print("\tSaving network in " + path)
    net.save(path)
    # get the results
    res = net.sim(input_data)

    return net, res

# 70 10 20
def divide_data_set(data_set):
    set_len = len(data_set)
    first_stop = int(math.floor(set_len * 0.7))
    second_stop = int(math.floor(set_len * 0.8))
    train_set = data_set[0:first_stop]
    validation_set = data_set[first_stop:second_stop]
    test_set = data_set[second_stop:-1]

    return train_set, validation_set, test_set


def test(x, y, net):
    netOutput = net.sim(x)  # simulate network, netOutput is y values

    existence = 0  # count number of right guesses

    for i in range(0, len(netOutput)):
        if y[i][0] == 1 and abs(netOutput[i][0]) > abs(netOutput[i][1]):
            existence += 1
        elif y[i][0] == 0 and abs(netOutput[i][0]) < abs(netOutput[i][1]):
            existence += 1
    print("\t\tPercentage of correctly recognized Parkinson using NN is: {0}%" \
          .format((existence / len(x)) * 100))
