import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from deepLearning.session_two.tf_untils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #只显示warning和error


np.random.seed(1)

# y_hat = tf.constant(36, name='y_hat')
# y = tf.constant(39, name='y')
# loss = tf.Variable((y - y_hat)**2, name='loss')
# init = tf.global_variables_initializer()
# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))
# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a,b)
# print(c)
# session = tf.Session()
# print(session.run(c))
# x = tf.placeholder(tf.int64, name='x')
# print(session.run(2*x, feed_dict={x:3}))
# session.close()

def linear_function():
    np.random.seed(1)

    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.constant(np.random.randn(4,3), name='W')
    b = tf.constant(np.random.randn(4,1), name='b')
    # Y = tf.add(tf.matmul(W, X), b)
    Y = tf.matmul(W,X) + b

    sess = tf.Session()
    result = sess.run(Y)
    sess.close()

    return result

def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})
    return result

def cost(logits, labels):

    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    sess = tf.Session()
    cost = sess.run(cost,feed_dict={z:logits,y:labels})
    sess.close()

    return cost

def one_hot_matrix(labels, C):
    C = tf.constant(C,name='C')
    one_hot_matrix = tf.one_hot(labels,C,axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    return ones

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    return X,Y

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25,12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25,1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12,1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6,1], initializer=tf.zeros_initializer())

    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2, "W3":W3, "b3":b3}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.matmul(W1,X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2,A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3,A2) + b3

    return Z3

def compute_cost(Z3, Y):
    # tf.transpose(a, perm = None, name = 'transpose')
    # 将a进行转置，并且根据perm参数重新排列输出维度。这是对数据的维度的进行操作的形式。
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True):
    tf.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iteration (per tens)")
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X:X_train, Y:Y_train}))
        print("Train Accuracy:", accuracy.eval({X:X_test, Y:Y_test}))

        return parameters

predict()