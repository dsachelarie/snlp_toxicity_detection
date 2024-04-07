from utils import read_data, get_igm_weights, add_embeddings, balance_dataset, get_embeddings_only


class Model:
    def __init__(self, mode="debug", preprocessing="glove", separate_word_embeddings=False):
        self.mode = mode
        igm_per_word = None

        train_data = read_data("data/train_2024.csv")

        if preprocessing == "glove_rtf_igm":
            igm_per_word = get_igm_weights(train_data)

        train_data = add_embeddings(train_data, igm_weights=igm_per_word, separate_word_embeddings=separate_word_embeddings)
        train_data = balance_dataset(train_data)

        if self.mode == "debug":
            test_data = read_data("data/dev_2024.csv")

        elif self.mode == "release":
            test_data = read_data("data/test_2024.csv")

        else:
            raise Exception(f"Mode \"{self.mode}\" is not supported!")

        test_data = add_embeddings(test_data, igm_weights=igm_per_word, separate_word_embeddings=separate_word_embeddings)

        self.X_train, self.y_train = get_embeddings_only(train_data)
        self.X_test, self.y_test = get_embeddings_only(test_data)
