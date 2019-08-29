import sys

try:
    import numpy as np
except ImportError:
    assert False, 'Error in import modules: %s' % sys.argv[0].split("/")[-1]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_class_names(directory_path):
    raw = unpickle(directory_path + '/meta')
    return raw[b'fine_label_names']


def get_train_data(directory_path):
    train = unpickle(directory_path + '/train')
    data = train[b'data'].reshape(-1, 3, 32, 32)
    labels = np.array(train[b'fine_labels'])
    return data, labels


def get_test_data(directory_path):
    test = unpickle(directory_path + '/test')
    data = test[b'data'].reshape(-1, 3, 32, 32)
    labels = np.array(test[b'fine_labels'])
    return data, labels


def shufffle(X, y):
    shuffled_index = np.random.permutation(X.shape[0])
    return X[shuffled_index], y[shuffled_index]


def train_val_split(X, y, ratio):
    idx = int(X.shape[0]*(1 - ratio))
    return X[:idx], X[idx:], y[:idx], y[idx:]


def make_batch(X, batch_size, type):
    B = batch_size
    if type == 'X':
        N, C, H, W = X.shape
        shape = (N // B, B, C, H, W)
        strides = (B * C * H * W, C * H * W, H * W, W, 1)
        strides = X.itemsize * np.array(strides)
        X = np.lib.stride_tricks.as_strided(X, shape, strides)
    elif type == 'y':
        N = X.shape[0]
        shape = (N // B, B)
        strides = (B, 1)
        strides = X.itemsize * np.array(strides)
        X = np.lib.stride_tricks.as_strided(X, shape, strides)
    return X


def get_cifar100_data(directory_path, validation_ratio=0.2, shuffle=False):
    X, y = get_train_data(directory_path)
    if shuffle:
        X, y = shufffle(X, y)

    X_train, X_val, y_train, y_val = train_val_split(X, y, validation_ratio)
    B = 32

    X_train, y_train = make_batch(X_train, B, 'X'), make_batch(y_train, B, 'y')
    X_test, y_test = get_test_data(directory_path)

    return X_train, X_val, X_test, y_train, y_val, y_test
