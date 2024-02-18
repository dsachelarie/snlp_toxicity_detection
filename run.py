from models.basic_model import BasicModel


if __name__ == "__main__":
    model = BasicModel(mode="debug", preprocessing="glove")
    model.run()
