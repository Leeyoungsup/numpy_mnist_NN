import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_node, hidden_nodes, output_node):
        self.input_node = input_node
        self.hidden_nodes = hidden_nodes
        self.output_node = output_node

        self.__W2 = np.random.rand(self.input_node, hidden_nodes)
        self.__b2 = np.random.rand(self.hidden_nodes)
        self.__W3 = np.random.rand(self.hidden_nodes, output_node)
        self.__b3 = np.random.rand(output_node)
        self.__learning_rate = 1e-4

    def feed_forward(self):
        delta = 1e-7
        z1 = np.dot(self.input_data, self.__W2) + self.__b2
        y1 = sigmoid(z1)
        z2 = np.dot(y1, self.__W3) + self.__b3
        y = sigmoid(z2)
        return -np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log(1 - y + delta))

    def loss_val(self):
        delta = 1e-7
        z1 = np.dot(self.input_data, self.__W2) + self.__b2
        y1 = sigmoid(z1)
        z2 = np.dot(y1, self.__W3) + self.__b3
        y = sigmoid(z2)
        return -np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log(1 - y + delta))

    def train(self, training_data):
        self.target_data = np.zeros(self.output_node) + 0.01
        self.target_data[int(training_data[0])] = 0.99
        self.input_data = (training_data[1:] / 255.0 * 0.99 + 0.01)
        f = lambda x: self.loss_val()
        self.__W2 -= self.__learning_rate * numerical_derivative(f, self.__W2)
        self.__b2 -= self.__learning_rate * numerical_derivative(f, self.__b2)
        self.__W3 -= self.__learning_rate * numerical_derivative(f, self.__W3)
        self.__b3 -= self.__learning_rate * numerical_derivative(f, self.__b3)

    def predict(self, input_data):
        z1 = np.dot(input_data, self.__W2) + self.__b2
        y1 = sigmoid(z1)
        z2 = np.dot(y1, self.__W3) + self.__b3
        y = sigmoid(z2)
        predicted_num = np.argmax(y)
        return predicted_num

    def accuracyTest(self, test_data):
        matched_count = 0
        not_matched_count = 0
        for index in range(len(test_data)):
            label = int(test_data[index, 0])
            data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
            predicted_num = self.predict(data)
            if label == predicted_num:
                matched_count += 1
            else:
                not_matched_count += 1
        print("Current Accuracy = ", 100 * (matched_count / len(test_data)), "%")
        return 100 * (matched_count / len(test_data))


def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


training_data = np.loadtxt('mnist_data/mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('mnist_data/mnist_test.csv', delimiter=',', dtype=np.float32)
input_nodes = training_data[:, 1:].shape[1]
hidden_nodes = 100
output_nodes = 10
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
for step in range(30001):
    nn.train(training_data[step])
    print("step = ", step, " loss_val = ", nn.loss_val())
