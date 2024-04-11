from models.basic_model import BasicModel
from models.bert_model import BertModel
from models.nn_model import NNModel


if __name__ == "__main__":
    model = BasicModel(mode="release", preprocessing="glove_rtf_igm")
    # model = BertModel(preprocessing="glove_rtf_igm")
    # model = NNModel(mode="debug", preprocessing="glove_rtf_igm")

    model.run()
