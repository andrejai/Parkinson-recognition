import csv
import os
from util import create_nn, denormalize_data, normalize_data, divide_data_set
import neurolab as nl

def read(path="\datasets\\3"):
    X = []
    Y = []

    for filename in os.listdir(os.getcwd()+path+"\hw_dataset\control"):
        if filename.endswith(".txt"):
            person = []
            f = open(os.getcwd()+path+"\hw_dataset\control\\"+filename, 'r')
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                person_data = []
                person_data.append([[float(x) for x in y.split(';')] for y in row])
                person.append(person_data)
            X.append(person)
            Y.append([0, 1])
    for filename in os.listdir(os.getcwd()+path+"\hw_dataset\parkinson"):
        if filename.endswith(".txt"):
            person = []
            f = open(os.getcwd()+path+"\hw_dataset\parkinson\\"+filename, 'r')
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                person_data = []
                person_data.append([[float(x) for x in y.split(';')] for y in row])
                person.append(person_data)
            X.append(person)
            Y.append([1, 0])
    for filename in os.listdir(os.getcwd()+path+"\\new_dataset\parkinson"):
        if filename.endswith(".txt"):
            f = open(os.getcwd()+path+"\\new_dataset\parkinson\\"+filename, 'r')
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                person_data = []
                person_data.append([[float(x) for x in y.split(';')] for y in row])
                person.append(person_data)
            X.append(person);
            Y.append([1, 0]);
    return X, Y

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

