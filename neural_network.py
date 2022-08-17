import numpy as np


def sigmoid(z):
    z = np.clip(z, -1000, 1000)
    exp = np.exp(-z)
    return 1 / (1 + exp)


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


class NeuralNetwork:
    def __init__(self, shape, epsilon_init=0.12):
        self.weights = []
        for i in range(len(shape) - 1):
            self.weights.append(np.random.rand(shape[i + 1], shape[i] + 1) * 2 * epsilon_init - epsilon_init)

    def feedforward(self, h):
        m = h.shape[0]
        for theta in self.weights:
            a = np.concatenate((np.ones((m, 1)), h), 1)
            h = sigmoid(np.matmul(a, np.transpose(theta)))
        return h

    def cost(self, X, y):
        m = X.shape[0]
        h = self.feedforward(X)
        cost = np.multiply(-y, np.log(h)) - np.multiply(1 - y, np.log(1 - h))
        cost = np.sum(cost) / m
        return cost

    def gradient(self, X, y):
        m = X.shape[0]
        gradients = []
        a = []
        z = []
        a_ = np.concatenate((np.ones((m, 1)), X), 1)
        for theta in self.weights:
            a.append(a_)
            z_ = np.matmul(a_, np.transpose(theta))
            z.append(z_)
            a_ = np.concatenate((np.ones((m, 1)), sigmoid(z_)), 1)
        a.append(sigmoid(z_))
        delta = [np.transpose(a[-1] -y)]
        for theta in reversed(self.weights[1:]):
            z.pop()
            delta.insert(0, np.multiply(np.matmul(np.transpose(theta), delta[0])[1:],
                                        sigmoid_gradient(np.transpose(z[0]))))
        for theta in self.weights:
            gradients.append(np.matmul(delta[0], a[0]) / m)
            a.pop(0)
            delta.pop(0)

        return gradients

    def train(self, X_train, y_train, batches, learning_rate, iterations):
        batch_size = np.floor(X_train.shape[0] / batches)
        for iteration in range(iterations):
            batch = iteration % batches
            start = int(batch * batch_size)
            end = int((batch + 1) * batch_size)
            X, y = X_train[start:end], y_train[start:end]
            gradients = self.gradient(X, y)
            for i in range(len(self.weights)):
                self.weights[i] -= gradients[i] * learning_rate

    def accuracy(self, X, y):
        pred = self.feedforward(X)
        pred = np.argmax(pred, 1)
        y = np.argmax(y, 1)
        m = X.shape[0]
        correct = 0
        for i in range(m):
            if pred[i] == y[i]:
                correct += 1
        return correct / m
