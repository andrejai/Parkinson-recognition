import os
import neurolab as nl
import numpy as np
import math
import pylab as pl
from sklearn.preprocessing import MinMaxScaler
from neurolab.trans import SoftMax, TanSig
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, \
    chi2
from sklearn.neural_network import MLPRegressor


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


def create_neural_network(input_data, target_data, minmax, path, hidden_layers=7, epochs=800):
    # init the neural network
    if os.path.isfile(path):
        print("\n\tLoading network from " + path)
        net = nl.load(path)
        res = net.sim(input_data)
        return net, res

    net = nl.net.newff(minmax, [hidden_layers, len(target_data[0])])
    # net.transf = nl.net.trans.SoftMax
    net.trainf = nl.net.train.train_rprop
    # net.errorf = nl.net.error.CEE
    # net.layers[-1].transf = nl.trans.SoftMax()
    net.init()
    # train the neural network
    err = net.train(input_data, target_data, epochs=epochs, show=10, goal=1)
    pl.figure(1)
    pl.plot(err)
    pl.show()
    print("\tSaving network in " + path)
    net.save(path)
    # get the results
    res = net.sim(input_data)

    return net, res


def denormalize_data(input_data, means, sqrt):
    denormalized = [[x * sqrt[y.index(x)] + means[y.index(x)] for x in y] for y in input_data]
    return denormalized


def create_nn(input_data, output_data, path, hidden_layers=8, epochs=500, goal=0.5):
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


def get_data(x_data, y_data):
    reduces = [y[0] for y in y_data]
    fs = feature_selection(np.array(x_data), np.array(reduces))
    n_x_data, means_x, sqrt_x = normalize_data(fs.tolist())
    n_y_data, means, sqrt = normalize_data(y_data)

    x_train, x_validation, x_test = divide_data_set(n_x_data)
    y_train, y_validation, y_test = divide_data_set(n_y_data)
    return x_train, x_validation, x_test, y_train, y_validation, y_test, means, sqrt
