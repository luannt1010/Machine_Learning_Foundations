import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(test_size=0.3):
    df = pd.read_csv("Iris.csv")
    X = df.drop(["Species", "Id"], axis=1)
    y = np.array(df["Species"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.reshape(1, len(y_train))
    y_test = y_test.reshape(1, len(y_test))
    return X_train, X_test, y_train, y_test, df

def transform_y(y):
    for i in range(len(y[0])):
        if y[0][i] == 'Iris-setosa':
            y[0][i] = 0
        elif y[0][i] == 'Iris-versicolor':
            y[0][i] = 1
        else:
            y[0][i] = 2
    return y