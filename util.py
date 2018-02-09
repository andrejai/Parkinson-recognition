import os
import neurolab as nl
import numpy as np
import math


def feature_selection(input_data):
    return


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

    return normalized


def create_nn(input_data, output_data, path, hidden_layers=30, epochs=100):
    # init the neural network
    net = nl.net.newff([[-1, 1]] * len(input_data[0]), [hidden_layers, len(output_data[0])])
    net.trainf = nl.net.train.train_gdx
    net.init()
    # train the neural network
    net.train(input_data, output_data, epochs=epochs, show=10, goal=0.01)
    # save network to a file
    print("\tSaving network in " + path)
    net.save(path)
    # get the results
    res = net.sim(input_data)

    return net, res
