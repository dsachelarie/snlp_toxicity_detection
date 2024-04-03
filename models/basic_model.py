from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from utils import read_data, write_preds, get_rtf_igm_weights, read_prob_weights_cached, get_rtf_igm_test_weights
import os


class BasicModel(Model):
    def __init__(self, mode="debug", preprocessing="glove_rtf_igm"):
        super(BasicModel, self).__init__(mode, preprocessing)


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
