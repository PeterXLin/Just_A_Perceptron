import numpy as np


class Dataset:
    def __init__(self, path, classes: int):
        raw_data = np.loadtxt(path, dtype=np.float, delimiter=' ')
        self.classes = classes
        self.feature = raw_data.shape[1] - 1
        np.random.shuffle(raw_data)
        training_data_size = (raw_data.shape[0] * 2) // 3
        # print(training_data_size)
        self.training_dataset = raw_data[:training_data_size]
        self.validation_dataset = raw_data[training_data_size:]
        self.training_data_size = training_data_size
        self.validation_data_size = raw_data.shape[0] - training_data_size


def get_class(type_index: int, output_dim):
    binary_represent = list()
    while type_index:
        binary_represent.append(type_index % 2)
        type_index = type_index // 2
    while len(binary_represent) < output_dim:
        binary_represent.append(0)
    return binary_represent


class Model:
    def __init__(self, input_dim: int, types: int, lr_rate: int):
        # need at least neuron_amount digit to present all the types
        neuron_amount = (types + 1) // 2
        neuron_list = list()
        for i in range(neuron_amount):
            weights = np.random.random(input_dim + 1)
            neuron_list.append(weights)

        self.input_dim = input_dim
        self.lr_rate = lr_rate
        self.neuron_list = neuron_list
        self.best_neuron_list = neuron_list.copy()
        self.neuron_amount = neuron_amount

    def update(self, feature):
        correct_output = get_class(feature[-1] - 1, self.neuron_amount)
        for i in range(self.neuron_amount):
            output_tmp = -1 * self.neuron_list[i][0]
            for j in range(self.input_dim):
                output_tmp = output_tmp + feature[j]*self.neuron_list[i][j+1]

            if output_tmp >= 0:
                output_tmp = 1
            else:
                output_tmp = 0

            if output_tmp != correct_output[i]:
                if output_tmp == 0:
                    self.neuron_list[i][0] = self.neuron_list[i][0] - self.lr_rate
                    for j in range(self.input_dim):
                        self.neuron_list[i][j+1] = self.neuron_list[i][j+1] + self.lr_rate * feature[j]
                else:
                    self.neuron_list[i][0] = self.neuron_list[i][0] + self.lr_rate
                    for j in range(self.input_dim):
                        self.neuron_list[i][j + 1] = self.neuron_list[i][j + 1] - self.lr_rate * feature[j]

    def predict(self, feature):
        """give feature return predict"""
        result = 0
        for i in range(self.neuron_amount):
            output_tmp = -1 * self.neuron_list[i][0]
            for j in range(self.input_dim):
                output_tmp = output_tmp + feature[j]*self.neuron_list[i][j+1]
            if output_tmp >= 0:
                output_tmp = 1
            else:
                output_tmp = 0
            result = result*2 + output_tmp
        return result + 1


def train_model(config: dict, dataset: Dataset, perceptron: Model):
    min_error = 10000
    check_count = 0
    for n in range(config['epoch']):
        np.random.shuffle(dataset.training_dataset)
        for training_data in dataset.training_dataset:
            perceptron.update(training_data)
            # check if model is better
            if check_count == config['check_packet_frequency']:
                tmp_error = 0
                for validation_data in dataset.validation_dataset:
                    if perceptron.predict(validation_data) != validation_data[-1]:
                        tmp_error = tmp_error + 1
                if tmp_error < min_error:
                    min_error = tmp_error
                    perceptron.best_neuron_list = perceptron.neuron_list.copy()
                    if tmp_error/dataset.validation_data_size <= config['error_rate']:
                        return perceptron
                check_count = 0
            else:
                check_count = check_count + 1
    return perceptron
