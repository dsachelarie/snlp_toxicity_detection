from models.basic_model import BasicModel


if __name__ == "__main__":
    model = BasicModel(mode="release", preprocessing="glove_rtf_igm")
    model.run()
