from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from utils import read_data, write_preds, get_tf_prob_weights


class BasicModel:
    def __init__(self, mode="debug", preprocessing="glove"):
        self.mode = mode
        weights = None

        if preprocessing == "glove_tf_prob":
            weights = get_tf_prob_weights("data/train_2024.csv")

        if self.mode == "debug":
            X, y = read_data("data/train_2024.csv", preprocessing, weights)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1)

        elif self.mode == "release":
            self.X_train, self.y_train = read_data("data/train_2024.csv", preprocessing, weights)
            self.X_test, self.y_test = read_data("data/test_2024.csv", preprocessing, weights)

        else:
            raise Exception(f"Mode \"{self.mode}\" is not supported!")

    def run(self):
        print("Running SVM")

        svm = SVC()
        svm.fit(self.X_train, self.y_train)
        preds = svm.predict(self.X_test)

        if self.mode == "debug":
            print(f1_score(preds, self.y_test))

        elif self.mode == "release":
            write_preds("data/preds.csv", preds)

        else:
            raise Exception(f"Mode \"{self.mode}\" is not supported!")
