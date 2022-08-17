import math
import numpy as np
from neural_network import NeuralNetwork
import gzip
import pickle
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt

EPOCHS = 100


def load_mnist():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='iso-8859-1')
    f.close()
    y_train, y_valid, y_test = np.array(train_set[1]), np.array(valid_set[1]), np.array(test_set[1])
    X_train, X_valid, X_test = np.array(train_set[0]), np.array(valid_set[0]), np.array(test_set[0])
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train, y_valid, y_test = lb.transform(y_train), lb.transform(y_valid), lb.transform(y_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def disc_training_data(X_train, n):
    idx = np.random.randint(1000, size=n)
    disc_X_train_real = X_train[idx, :]
    disc_y_train_real = np.ones((n, 1))
    noise = np.random.random((n, 4))
    disc_X_train_gen = generator.feedforward(noise)
    disc_y_train_gen = np.zeros((n, 1))
    disc_X_train = np.empty((2 * n, 2))
    disc_y_train = np.empty((2 * n, 1))
    disc_X_train[0::2] = disc_X_train_real
    disc_X_train[1::2] = disc_X_train_gen
    disc_y_train[0::2] = disc_y_train_real
    disc_y_train[1::2] = disc_y_train_gen
    return disc_X_train, disc_y_train


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # X_train, X_valid, X_test, y_train, y_valid, y_test = load_mnist()

    filename = 'discriminator.p'
    try:
        infile = open(filename, 'rb')
        discriminator = pickle.load(infile)
        infile.close()
    except:
        discriminator = NeuralNetwork([2, 4, 1])

    filename = 'generator.p'
    try:
        infile = open(filename, 'rb')
        generator = pickle.load(infile)
        infile.close()
    except:
        generator = NeuralNetwork([4, 4, 2])

    train_data_length = 1000
    train_data = np.zeros((train_data_length, 2))
    train_data[:, 0] = np.random.random(train_data_length)
    train_data[:, 1] = np.sin(2 * math.pi * train_data[:, 0])

    for epoch in range(EPOCHS):
        n = 100
        disc_X_train, disc_y_train = disc_training_data(train_data, n)
        discriminator.train(disc_X_train, disc_y_train, 200, 0.1, 100)
        noise = np.random.random((n, 4))
        generated_samples = generator.feedforward(noise)
        y_train = np.zeros((n, 1))
        generator.train(noise, y_train, 200, 0.1, 100)
        if epoch % 10 == 0:
            print(epoch)
            noise = np.random.random((100, 4))
            data = generator.feedforward(noise)
            plt.plot(data[:, 0], data[:, 1], ".")
            plt.show()


