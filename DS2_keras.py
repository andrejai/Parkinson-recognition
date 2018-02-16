import csv
import os

import math
from keras.models import Sequential, model_from_json
from keras.layers import Dense, np, Dropout
from keras.optimizers import SGD, RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from random import shuffle
from util import feature_selection, normalize_data, divide_data_set, find_minmax, denormalize_data


def read(file_name):
    print("\t\tReading data from file...")
    X = []
    Y = []
    with open(file_name, 'rt') as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = []

        for row in reader:
            rows.append(row)

        shuffle(rows)
        for row in rows:
            Y.append([float(_x) for _x in row[4:6]])
            del row[4:6]
            del row[0]
            del row[2]
            X.append([int(_y) if _y.isdigit() else float(_y) for _y in row])

    return X, Y


def create_model():
    model = Sequential()

    model.add(Dense(units=8, activation='relu', input_dim=10))
    model.add(Dropout(0.2))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='relu'))

    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def train():
    x_data, y_data = read(os.getcwd() + "\datasets\\2\parkinsons_updrs.data.csv")

    reduces = [y[0] for y in y_data]
    x_data = feature_selection(np.array(x_data), np.array(reduces)).tolist()

    x_data, m, s = normalize_data(x_data)

    x_train, x_validation, x_test = divide_data_set(x_data)
    y_train, y_validation, y_test = divide_data_set(y_data)

    motor_x_train = np.array([[y[0]] for y in y_train])
    total_x_train = np.array([[y[1]] for y in y_train])
    motor_x_test = np.array([[y[0]] for y in y_test])
    total_x_test = np.array([[y[1]] for y in y_test])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("\t\tFinished divide data...")
    n_x_train = np.array(x_train)
    n_y_train = np.array([[y[0]] for y in y_train])
    n_y_train = np.array(y_train)
    print("\t\tTraining neural network...")
    # Step 6: train network
    path = input("Input output file name for neural network(enter for end): ")
    while path != "":
        for i in range(0, 10):
            net = create_neural_network(n_x_train, n_y_train, path=os.getcwd() + "\\" + path, lr=(i + 1) / 10)
            test(x_test, y_test, net, i)
        path = input("Input output file name for neural network(enter for end): ")


def train_reg(X, Y, fn, X_test, Y_test, seed=7):
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=fn, epochs=100, batch_size=128, verbose=0)
    kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(pipeline, X, Y, cv=kfold)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print(results)
    print('Result: %.2f (%.2f) MSE' % (results.mean(), results.std()))

    estimator.fit(X, Y)
    netOutput = estimator.predict(X_test)

    print("Loss and metrics")
    print(rmse(netOutput, Y_test))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def test(x_test, y_test, net, i=0):
    # simulate network, netOutput is y values
    # loss_and_metrics = net.evaluate(x_test, y_test, batch_size=150)
    results = net.predict(x_test)
    print("Results ")
    print(results)
    print(y_test)

    print("Loss and metrics for :" + str(i))
    print(rmse(results, y_test))


def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path + ".model.h5")
    print("Saved model to disk")


def load_model(path):
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + "model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='mean_squared_error',
                         optimizer='sgd',
                         metrics=['accuracy'])
    return loaded_model


def create_neural_network(x_train, y_train, path, lr=0.1, epochs=50, batch=50):
    if os.path.isfile(path):
        return load_model(path)

    model = Sequential()

    model.add(Dense(units=8, activation='relu', input_dim=len(x_train[0])))
    model.add(Dropout(0.1))
    model.add(Dense(units=8, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(units=8, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(units=len(y_train[0]), activation='relu'))

    sgd = SGD(lr=lr, momentum=0.1, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mae'])

    model.fit(np.array(x_train), np.array(y_train), epochs=epochs, batch_size=batch, verbose=0)
    # serialize model to JSON
    save_model(model, path)

    net = model

    return net


def test_sgd_regressor(x_data, y_data, x_test, y_test, alpha=0.1):
    clf = linear_model.SGDRegressor(shuffle=True, tol=0.2, loss='squared_loss', alpha=alpha)
    clf.fit(x_data, y_data)
    result = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("Loss and metrics")
    print(rmse(result, y_test))
    print(score)
    print(result)


def test_lasso_regressor(x_data, y_data, x_test, y_test, alpha=0.1):
    clf = clf = linear_model.Lasso(alpha=alpha)
    clf.fit(x_data, y_data)
    result = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("Loss and metrics")
    print(rmse(result, y_test))
    print(score)
    print(result)
