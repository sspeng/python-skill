import numpy as np
import time
import copy
# m_train = train_set_x_orig.shape[0]
# m_test = test_set_x_orig.shape[0]
# num_px = train_set_x_orig.shape[1]

def initialize_with_zeros(dim):
    w = np.zeros([dim, 1])
    b = 0
    return [w,b]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T))/m
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    assert dw.shape == w.shape
    assert  db.dtype == float
    cost = np.squeeze(cost)
    assert cost.shape == ()
    grads = {"dw": dw, "db": db}
    return [grads, cost]


def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i%100==0:
            costs.append(cost)
        if print_cost and i%100==0:
            print("Cost after iteration %i:%f" % (i,cost))
    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    return [params,grads,costs]


def predict(w,b,X):
    m = X.shape[1]
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    Y_prediction = A
    Y_prediction[Y_prediction>=0.5] = 1
    Y_prediction[Y_prediction<0.5] = 0
    assert Y_prediction.shape == (1,m)
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations=num_iterations,learning_rate=learning_rate,print_cost=print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_train)
    Y_prediction_train = predict(w, b, X_test)
    print("train accuracy:{}%".format(100 - np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy:{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test))*100))
    print("costs:{}%".format(costs))

# w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1,0,1]])
# params,grads,cost = optimize(w,b,X,Y,num_iterations=100,learning_rate=0.009,print_cost=False)
# print(params,grads)
# grads, cost = propagate(w, b, X, Y)
# print(grads, cost)
# print(initialize_with_zeros(2))
# w = np.array([[0.1124579],[0.23106775]])
# b = -0.3
# X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
# print(predict(w,b,X))