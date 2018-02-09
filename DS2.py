import csv
import math


def read(file_name="parkinsons_udprs.data.csv"):
    X = []
    Y = []
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            Y.append([float(_x) for _x in row[4:6]])
            del row[4:6]
            X.append([int(_y) if _y.isdigit() else float(_y) for _y in row])
    return X, Y


# 70 10 20
def divide_data_set(data_set):
    set_len = len(data_set)
    first_stop = int(math.floor(set_len * 0.7))
    second_stop = int(math.floor(set_len * 0.8))
    train_set = data_set[0:first_stop]
    validation_set = data_set[first_stop:second_stop]
    test_set = data_set[second_stop:-1]

    return train_set, validation_set, test_set
