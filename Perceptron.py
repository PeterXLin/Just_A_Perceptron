import numpy as np
import matplotlib.pyplot as plt


# read data
# data class
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


def get_data_distribution_map_and_line(path, weight: list):
    data_all = np.loadtxt(path, dtype=np.float, delimiter=' ')
    # data distribution scatter
    x_1 = list()
    x_2 = list()
    y_1 = list()
    y_2 = list()
    for d in data_all:
        if d[2] == 1:
            x_1.append(d[0])
            y_1.append(d[1])
        elif d[2] == 2:
            x_2.append(d[0])
            y_2.append(d[1])
    plt.scatter(x_1, y_1, c='red', s=5)
    plt.scatter(x_2, y_2, c='green', s=5)
    # model prediction line
    x1_min = np.amin(data_all, axis=0)[0]
    x1_max = np.amax(data_all, axis=0)[0]
    x = np.arange(x1_min, x1_max, 0.1)
    # the divide line
    y = -(-weight[0] + weight[1] * x) / weight[2]
    plt.plot(x, y, c='purple')
    # other setting
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Data Distribution Map')
    plt.show()


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


np.random.seed(1)
# validation
config = {'path': 'Dataset/2Hcircle1.txt',
          'classes': 2,
          'learning_rate': 0.8,
          'check_size': 30,
          'epoch': 300}


# training
dataset = Dataset(config['path'], config['classes'])
perceptron = Model(dataset.feature, config['classes'], config['learning_rate'])
min_error = 10000
check_count = 0
for n in range(config['epoch']):
    np.random.shuffle(dataset.training_dataset)
    for training_data in dataset.training_dataset:
        perceptron.update(training_data)

        # check if model is better
        if check_count == config['check_size']:
            tmp_error = 0
            for validation_data in dataset.validation_dataset:
                if perceptron.predict(validation_data) != validation_data[-1]:
                    tmp_error = tmp_error + 1
            if tmp_error < min_error:
                min_error = tmp_error
                perceptron.best_neuron_list = perceptron.neuron_list.copy()
            check_count = 0
        else:
            check_count = check_count + 1
print(perceptron.best_neuron_list[0])
get_data_distribution_map_and_line(config['path'], perceptron.best_neuron_list[0])
