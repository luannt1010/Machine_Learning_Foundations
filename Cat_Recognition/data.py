import h5py
import numpy as np

def load_dataset(train_path, test_path):
    train_dataset = h5py.File(train_path, 'r')
    train_set_x = train_dataset['train_set_x'][:] # nparr (209, 64, 64, 3)
    train_set_y = train_dataset['train_set_y'][:]
    
    test_dataset = h5py.File(test_path, 'r')
    test_set_x = test_dataset['test_set_x'][:]
    test_set_y = test_dataset['test_set_y'][:]

    classes = test_dataset["list_classes"][:]
    
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
    return train_set_x, test_set_x, train_set_y, test_set_y, classes

# train_path = r'dataset/train_catvsnoncat.h5'
# test_path = r'dataset/test_catvsnoncat.h5'
# X_train, X_test, Y_train, Y_test = load_dataset(train_path, test_path)
# print(X_train.shape)
# print(Y_train.shape)