import os
import neurolab as nl
import numpy as np
import math
from sklearn.feature_selection import SelectKBest, f_classif


def feature_selection(data, target):
    new_input = SelectKBest(f_classif, k=10)
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

def get_data(x_data, y_data):
    reduces = [y[0] for y in y_data]
    fs = feature_selection(np.array(x_data), np.array(reduces))
    n_x_data, means_x, sqrt_x = normalize_data(fs.tolist())
    n_y_data, means, sqrt = normalize_data(y_data)

    x_train, x_validation, x_test = divide_data_set(n_x_data)
    y_train, y_validation, y_test = divide_data_set(n_y_data)
    return x_train, x_validation, x_test, y_train, y_validation, y_test, means, sqrt