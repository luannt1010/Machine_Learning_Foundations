import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data

X_train, X_test, y_train, y_test, df = load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train.shape)

# Pie chart for count each species
species_count = df["Species"].value_counts().to_list()
species_list = df["Species"].unique()
plt.pie(x=species_count, labels=species_list, autopct='%1.1f%%')
plt.show()

# Histogram for distribution for each feature
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
for feature in features:
    sns.histplot(data=df, x=feature, hue="Species", kde=True, palette="Set1", bins=50)
    plt.show()

class KNN:
    def __init__(self, k):
        self.k = k

    def euclid_distance(self, training_point, test_point):
        distance = np.sqrt((np.sum(np.square(training_point - test_point))))
        return distance

    def knn_predict(self, training_data, training_label, point, k=None):
        if k is None:
            k = self.k
        distances = []
        predictions = []
        for i in range(len(training_data)):
            dist = self.euclid_distance(training_data[i], point)
            distances.append((dist, training_label[0, i]))
        distances.sort(key=lambda x: x[0])
        for i in range(k):
            predictions.append(distances[i][-1])
        return predictions

    def get_final_answer(self, predictions):
        frequency = {}
        for label in predictions:
            if label not in frequency:
                frequency[label] = 1
            else:
                frequency[label] += 1
        return max(frequency, key=lambda x: frequency[x])

    def train(self, X_train, X_test, y_train, k=None):
        if k is None:
            k = self.k
        predicts = []
        for i in range(len(X_test)):
            test_point = X_test[i]
            predictions = self.knn_predict(X_train, y_train, test_point, k)
            predict = self.get_final_answer(predictions)
            predicts.append(predict)
        return predicts

    def get_accuracy(self, y_true, predicts):
        return accuracy_score(y_true, predicts)

    def optimize_k(self, X_train, X_test, y_train, y_test, loss=False):
        ks = [i for i in range(1, self.k + 1)]
        y_true = np.squeeze(y_test)
        best_k = {"k": [], "score": []}
        for k in ks:
            predicts = self.train(X_train, X_test, y_train, k)
            if not loss:
                score = self.get_accuracy(y_true, predicts)
            else:
                score = np.mean(y_true != predicts)
            best_k["k"].append(k)
            best_k["score"].append(score)
        return best_k

if __name__ == "__main__":
    knn = KNN(k=105)
    loss = "Loss"
    dic = knn.optimize_k(X_train, X_test, y_train, y_test, loss)
    k = dic["k"]
    score = dic["score"]
    label = None
    if not loss:
        idx_best = np.argmax(score)
        label = "Score"
    else:
        idx_best = np.argmin(score)
        label = "Loss"
    k_best = k[idx_best]
    score_best = score[idx_best]
    print(f"Best K: {k_best}")
    if loss:
        print(f"Best loss: {score_best}")
    else:
        print(f"Best score: {score_best}")
    plt.figure(figsize=(10, 6))
    plt.plot(k, score, color="blue", linestyle="-", linewidth=2, label=label)
    plt.scatter(k_best, score_best, color='red', s=50, label=f'Best K={k_best}, Score={score_best}')
    plt.annotate(f"Best K={k_best}\nScore={score_best}",
                 xy=(k_best, score_best), xytext=(k_best+0.5, score_best-0.02))
    plt.title(f"{label} at value K")
    plt.xlabel("K")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Predict unseen data
    data_test = np.array([[6.15, 3.05, 4.55, 1.35],
                          [4.85, 3.55, 1.45, 0.25],
                          [6.85, 3.05, 5.65, 2.05]])
    label_test = [["versicolor"],
                  ["setosa"],
                  ["virginica"]]
    for i in range(len(data_test)):
        predicts = knn.knn_predict(X_train, y_train, data_test[i], 5)
        pred = knn.get_final_answer(predicts)
        print(predicts)
        print(f"Actual: {np.squeeze(label_test[i])}, Predict: {pred}")

    # Output:
    # Actual: versicolor, Predict: Iris-versicolor
    # Actual: setosa, Predict: Iris-setosa
    # Actual: virginica, Predict: Iris-virginica

