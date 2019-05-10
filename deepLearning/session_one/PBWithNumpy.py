import math
import numpy as np


def basic_sigmoid(x):
    s = 1/(1+math.exp(-x))
    return s


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)


def image2vector(image):
    return image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)


def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x/x_norm

def softmax(x):
    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp/x_exp_sum


def L1(yhat, y):
    return np.sum(abs(y-yhat))


def L2(yhat, y):
    return np.sum(pow(yhat-y, 2))


# print(L2(np.array([.9, .2, .1, .4, .9]), np.array([1, 0, 0, 1, 1])))
# print(softmax(np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])))
# print(normalizeRows(np.array([[0, 3, 4], [1, 6, 4]])))
# print(sigmoid(np.array([0, 2])))
# print(image2vector(np.random.randn(3, 3, 3)*10))
