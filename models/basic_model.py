from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import read_data


class BasicModel:
    def __init__(self, mode="debug"):
        self.mode = mode

        if self.mode == "debug":
            X, y = read_data("data/train_2024.csv")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1)

        elif self.mode == "release":
            self.X_train, self.y_train = read_data("data/train_2024.csv")
            self.X_test, self.y_test = read_data("data/test_2024.csv")
            self.y_test = None

    def run(self):
        svm = SVC()
        svm.fit(self.X_train, self.y_train)
        preds = svm.predict(self.X_test)

        if self.mode == "debug":
            print(accuracy_score(preds, self.y_test))
