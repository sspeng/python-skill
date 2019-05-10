import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return [A, Z]
def relu(Z):
    A = np.maximum(0,Z)
    return [A, Z]
def sigmoid_backward(dA,activation_cache):
    A, Z = sigmoid(activation_cache)
    dZ = dA*(A*(1-A))
    return dZ
def relu_backward(dA,activation_cache):
    dZ = dA * (activation_cache > 0)
    return dZ


def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)

    W1 = np.random.randn(n_h,n_x)
    b1 = np.random.randn(n_h,1)
    W2 = np.random.randn(n_y,n_h)
    b2 = np.random.randn(n_y,1)

    paramters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return paramters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters["b"+str(l)] = np.random.randn(layer_dims[l],1)
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return [Z,cache]

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return [A,cache]

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    return [AL,cache]

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(1/m)*(np.dot(Y,np.log(AL).T)+np.dot(1-Y,np.log(1-AL).T))
    cost = np.squeeze(cost)
    assert cost.shape == ()
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    return [dA_prev,dW,db]

def linear_activation_backward(dA, cache, activation):
    if activation == "relu":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])
    elif activation == "sigmoid":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ. cache[0])

    return [dA_prev, dW, db]

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = AL - Y
    current_cache = caches[L-1]
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range[L-1]):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] + learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] + learning_rate*grads["db"+str(l+1)]

    return parameters
