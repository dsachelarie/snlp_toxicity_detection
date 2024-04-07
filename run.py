from models.basic_model import BasicModel
from models.bert_model import BertModel


if __name__ == "__main__":
    model = BasicModel(mode="release", preprocessing="glove_rtf_igm")
    # model = BertModel(preprocessing="glove_rtf_igm")
    # model = BertModel()

    model.run()
