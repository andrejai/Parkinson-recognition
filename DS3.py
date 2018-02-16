import csv
import os
from util import create_nn, denormalize_data, get_data_ds3
from random import shuffle


def read(path="\datasets\\3\data"):
    X = []
    Y = []
    files = os.listdir(os.getcwd()+path)
    shuffle(files)
    for filename in files:
        person = {0: [], 1: [], 2: []}
        f = open(os.getcwd()+path+"\\"+filename, 'r')
        for row in f.readlines():
            person[int(row.strip().split(';')[-1])].append([float(x) for x in row.strip().split(';')])
        X.append(person)
        for i in range(0, 3):
            if len(person[i]) != 0:
                if filename.startswith("C"):
                    Y.append([0, 1])
                else:
                    Y.append([1, 0])
    return X, Y


def generate_data(X):
    x_data = []
    for x in X:
        for i in range(0, 3):
            person = []
            if len(x[i]) > 0:
                angle = [item[3] for item in x[i]]
                pressure = [item[4] for item in x[i]]
                person.append(len(x[i]))
                person.append(sum(angle) / len(angle))
                person.append(sum(pressure) / len(pressure))
                person.append((x[i][-1][5] - x[i][0][5]) / 60)
                person.append(max(angle))
                person.append(min(angle))
                person.append(min(pressure))
                person.append(max(pressure))
                x_data.append(person)
    return x_data


def test(x, y, net):
    #simulate network, netOutput is y values
    netOutput = net.sim(x)

    #count number of right guesses
    existence = 0
    print(y)
    print(netOutput)
    for i in range(0, len(netOutput)):
        if y[i][0]== 1 and abs(netOutput[i][0]) > abs(netOutput[i][1]):
            existence += 1
        elif y[i][0] == 0 and abs(netOutput[i][0]) < abs(netOutput[i][1]):
            existence += 1
    print("\t\tPercentage of correctly recognized Parkinson using NN is: {0}%" \
          .format((existence / len(x)) * 100))


def train():
    x_data, y_data = read()
    x_data = generate_data(x_data)
    x_train, x_validation, x_test, y_train, y_validation, y_test = get_data_ds3(x_data, y_data)
    x_train = x_train + x_validation
    y_train = y_train + y_validation
    print(len(x_train))
    print(len(y_train))
    path = input("Input output file name for neural network(enter for end): ")
    while path != "":
        path = path + ".net"
        net, res = create_nn(x_train, y_train, os.getcwd() + "\\" + path, 7, 6000, 0.001)
        test(x_test, y_test, net)
        path = input("Input output file name for neural network(enter for end): ")
