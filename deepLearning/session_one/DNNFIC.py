import matplotlib.pyplot as plt
from scipy import ndimage,misc
from deepLearning.session_one.BYDNN import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x_orig = np.random.randn(10,20)
test_x_orig = np.random.randn(10,20)
index = 10
plt.show(train_x_orig[index])

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

train_x = train_x_flatten/255
test_x = test_x_flatten/255

n_x = 12288
n_h = 7
n_y = 1

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2)-np.divide((1-Y,1-A2)))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        if print_cost and i % 100 == 0:
            print("Cost after iteration  %i: %f" % (i, cost))
            costs.append(cost)

    def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        np.random.seed(1)
        costs = []
        parameters = initialize_parameters_deep(layers_dims)
        for i in range(num_iterations):
            AL, caches = L_model_forward(X, parameters)
            cost = compute_cost(AL, Y)
            grads = L_model_backward(AL, Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)

            if print_cost and i % 100 == 0:
                print("Cost after iteration  %i: %f" % (i, cost))
                costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def predict(my_image, my_label_y, parameters):
    pass

my_image = "my_image.jpg"
my_label_y = [1]

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False)) # 将图片转化为数组
my_image = misc.imresize(image, size=(num_px,num_px).reshape((num_px*num_px*3,1)))
my_image = my_image/255
parameters = None
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
