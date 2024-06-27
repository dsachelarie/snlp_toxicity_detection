from models.nn_model import NNModel


if __name__ == "__main__":
    model = NNModel(mode="debug", preprocessing="glove_rtf_igm")

    model.run()
