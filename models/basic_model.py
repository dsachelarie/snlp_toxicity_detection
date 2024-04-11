from models.model import Model
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from utils import write_preds


class BasicModel(Model):
    def __init__(self, mode="debug", preprocessing="glove_rtf_igm"):
        super(BasicModel, self).__init__(mode, preprocessing)

    def run(self):
        svm = SVC()
        svm.fit(self.X_train, self.y_train)
        preds = svm.predict(self.X_test)

        if self.mode == "debug":
            print(f1_score(preds, self.y_test))

        elif self.mode == "release":
            write_preds("data/preds_svm.csv", preds)

        else:
            raise Exception(f"Mode \"{self.mode}\" is not supported!")
