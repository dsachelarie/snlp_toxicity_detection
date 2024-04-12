from models.basic_model import BasicModel
from models.bert_model import BertModel
from utils import write_ensemble_preds


if __name__ == "__main__":
    #model = BasicModel(mode="debug", preprocessing="glove_rtf_igm")
    # model = BertModel(preprocessing="glove_rtf_igm")
    # model = BertModel()

    #model.run()

    # Aggregating ensemble predictions
    file_names = ["data/preds_roberta_5k_submitted.csv", "data/preds_alexnet.csv", "data/preds_svm.csv"]
    write_ensemble_preds(file_names)