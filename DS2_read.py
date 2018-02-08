import csv


def read(file_name="parkinsons_udprs.data.csv"):
    X = []
    Y = []
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            y = [float(_x) for _x in row[4:6]]
            print(y)
            Y.append(y)
            del row[4:6]
            x = [int(_y) if _y.isdigit() else float(_y) for _y in row]
            print(x)
            X.append(x)
