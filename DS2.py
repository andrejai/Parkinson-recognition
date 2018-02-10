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



