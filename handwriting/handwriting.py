import struct
import numpy as np
import matplotlib.pyplot as plt
# import os
from scipy.special import expit


def load_data():
    with open('train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('train-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    with open('t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('t10k-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    return train_images, train_labels, test_images, test_labels


def visualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(64):
        img = img_array[label_array == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()

# visualize_data(train_x, train_y)


def enc_one_hot(y, num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    for i, val in enumerate(y):
        one_hot[val, i] = 1.0
    return one_hot

# y = np.array([4, 5, 9, 0])
# z = enc_one_hot(y)
# print(y)
# print(z)


def sigmoid(z):
    # return(1/(1 + np.exp(-z)))
    return expit(z)


def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)


def visualize_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()

# visualize_sigmoid()


def calc_cost(y_enc, outpt):
    t1 = y_enc * np.log(outpt)
    t2 = (1 - y_enc)*np.log(1-outpt)
    cost = -np.sum(t1 + t2)
    return cost


def add_bias_unit(x, where):
    # where is just row or column
    if where == 'column':
        x_new = np.ones((x.shape[0], x.shape[1] + 1))
        x_new[:, 1:] = x
    elif where == 'row':
        x_new = np.ones((x.shape[0]+1, x.shape[1]))
        x_new[1:, :] = x
    return x_new


def init_weights(n_features, n_hidden, n_output):
    w1 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_features+1))
    w1 = w1.reshape(n_hidden, n_features+1)
    w2 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_hidden+1))
    w2 = w2.reshape(n_hidden, n_hidden+1)
    w3 = np.random.uniform(-1.0, 1.0, size=n_output*(n_hidden+1))
    w3 = w3.reshape(n_output, n_hidden+1)
    return w1, w2, w3


def feed_forward(x, w1, w2, w3):
    # add bias unit to the input
    # column wintin the row is just a byte of data
    # so we needd to add a column vector of ones
    a1 = add_bias_unit(x, where='column')
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)
    # since we transposed we have to add bias units as a row
    a2 = add_bias_unit(a2, where='row')
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)
    a3 = add_bias_unit(a3, where='row')
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4


def predict(x, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4 = feed_forward(x, w1, w2, w3)
    y_pred = np.argmax(a4, axis=0)
    return y_pred


def calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    delta4 = a4 - y_enc
    z3 = add_bias_unit(z3, where='row')
    delta3 = w3.T.dot(delta4)*sigmoid_gradient(z3)
    delta3 = delta3[1:, :]
    z2 = add_bias_unit(z2, where='row')
    delta2 = w2.T.dot(delta3)*sigmoid_gradient(z2)
    delta2 = delta2[1:, :]

    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)

    return grad1, grad2, grad3


def runModel(x, y, x_t, y_t):
    x_copy, y_copy = x.copy(), y.copy()
    y_enc = enc_one_hot(y)
    epochs = 50
    batch = 50

    w1, w2, w3 = init_weights(784, 75, 10)

    alpha = 0.001
    eta = 0.001
    dec = 0.00001
    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)

    for i in range(epochs):
        total_cost = []
        shuffle = np.random.permutation(y_copy.shape[0])
        x_copy, y_enc = x_copy[shuffle], y_enc[:, shuffle]
        eta /= (1 + dec*i)
        mini = np.array_split(range(y_copy.shape[0]), batch)

        for step in mini:
            # feed forward the model
            a1, z2, a2, z3, a3, z4, a4 = feed_forward(x_copy[step], w1, w2, w3)
            cost = calc_cost(y_enc[:, step], a4)

            total_cost.append(cost)
            # back propagate
            grad1, grad2, grad3 = calc_grad(a1, a2, a3, a4, z2, z3, z4,
                                            y_enc[:, step], w1, w2, w3)
            delta_w1, delta_w2, delta_w3 = \
                eta * grad1, eta * grad2, eta * grad3

            w1 -= delta_w1 + alpha * delta_w1_prev
            w2 -= delta_w2 + alpha * delta_w2_prev
            w3 -= delta_w3 + alpha * delta_w3_prev

            delta_w1_prev, delta_w2_prev, delta_w3_prev = \
                delta_w1, delta_w2, delta_w3

        print('epoch #', i)
    y_pred = predict(x_t, w1, w2, w3)
    acc = np.sum(y_t == y_pred, axis=0) / x_t.shape[0]
    print('training accuracy', acc*100)
    return 1


train_x, train_y, test_x, test_y = load_data()

t = runModel(train_x, train_y, test_x, test_y)
