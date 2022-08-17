import numpy as np
from neural_network import NeuralNetwork
import gzip
import pickle
from sklearn.preprocessing import LabelBinarizer


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_mnist()

    filename = 'neural_net.p'
    try:
        infile = open(filename, 'rb')
        nn = pickle.load(infile)
        infile.close()
    except:
        nn = NeuralNetwork([784, 400, 10])

    for i in range(5):
        nn.train(X_train, y_train, 4, 0.15, 20)
        accuracy = nn.accuracy(X_test, y_test) * 100
        print(f"Test set accuracy is {accuracy:.2f}%")
        outfile = open(filename, 'wb')
        pickle.dump(nn, outfile)
        outfile.close()
