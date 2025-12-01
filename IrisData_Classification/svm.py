import matplotlib.pyplot as plt
from load_data import load_data, transform_y
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_pram=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_pram = lambda_pram
        self.n_iters = n_iters
        self.total_loss = []
        self.hingeLoss = []
        self.w = None
        self.b = 0

    def initialize_weights(self, n_features):
        np.random.seed(42)
        self.w = np.random.rand(n_features)

    def label_y(self, Y):
        Y = np.where(Y <= 0, -1, 1)
        return Y

    def compute_distance(self, X_n):
        distance = np.dot(self.w, X_n) + self.b
        return distance

    def hinge_loss(self, X_n, Y_n):
        distance = self.compute_distance(X_n)
        condition = 1 - Y_n * distance #Must <= 0
        return max(0, condition)

    def gradient(self, X_n, Y_n):
        hinge_loss = self.hinge_loss(X_n, Y_n)
        gradients = {}
        if hinge_loss == 0:
            dw = 2 * self.lambda_pram * self.w
            db = 0
        else:
            dw = 2 * self.lambda_pram * self.w - np.dot(Y_n, X_n)
            db = -Y_n
        gradients["dw"] = dw
        gradients["db"] = db
        return gradients

    def update_parameters(self, X_n, Y_n):
        gradients = self.gradient(X_n, Y_n)
        dw = gradients["dw"]
        db = gradients["db"]
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def cost_function(self, X, y):
        losses = 0
        for i in range(len(X)):
            loss = self.hinge_loss(X[i], y[i])
            losses += loss
            self.hingeLoss.append(loss)
        cost = self.lambda_pram * np.sum(np.square(self.w)) + losses
        self.total_loss.append(cost)
        return cost

    def fit(self, X, y):
        y = self.label_y(y)
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                self.update_parameters(x_i, y[idx])

            cost = self.cost_function(X, y)
            print(f"Loss at epoch {epoch + 1} is: {cost}")

    def predict(self, point):
        distance = self.compute_distance(point)
        approx = np.sign(distance)
        return approx, distance

    def predict_test(self, X_test):
        preds = []
        for i in range(len(X_test)):
            pred, _ = self.predict(X_test[i])
            preds.append(pred)
        return preds

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = load_data(test_size=0.3)
    y_train = transform_y(y_train)
    y_test = transform_y(y_test)
    y_train = np.squeeze(np.where(y_train == 0, -1, 1))
    y_test = np.squeeze(np.where(y_test == 0, -1, 1))

    svm = SVM()
    svm.fit(X_train, y_train)

    data_test = np.array([[6.15, 3.05, 4.55, 1.35],
                          [4.85, 3.55, 1.45, 0.25],
                          [6.85, 3.05, 5.65, 2.05]])
    label_test = ["versicolor","setosa","virginica"]

    for i in range(len(data_test)):
        point = data_test[i]
        approx, d = svm.predict(point)
        print(approx, d)
    # Output: 1, -1, 1

    preds_test = svm.predict_test(X_test)
    accuracy_test = accuracy_score(y_test, preds_test)
    preds_train = svm.predict_test(X_train)
    accuracy_train = accuracy_score(y_train, preds_train)
    print("Accuracy on train set: ",accuracy_train)  # Accuracy on train = 1
    print("Accuracy on test set: ",accuracy_test) # Accuracy on test = 1

    total_loss = svm.total_loss
    plt.plot([i + 1 for i in range(1000)], total_loss, linestyle="-", linewidth=2, color="steelblue")
    plt.title("Loss at Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

