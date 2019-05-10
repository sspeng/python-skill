import tensorflow as tf
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from deepLearning.session_four.resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K
K.set_image_data_format("channels_last")
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):# TODO 与答案不符
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base+"2a", kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same", name=conv_name_base+"2b", kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1, 1), padding="valid", name=conv_name_base+"2c", kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)

    X = X + X_shortcut
    X = Activation("relu")(X)

    return X
# TODO 与答案不符
# tf.reset_default_graph()
# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder("float", [3, 4, 4, 6])
#     X = np.random.randn(3, 4, 4, 6)
#     A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block="a")
#     test.run(tf.global_variables_initializer())
#     out = test.run([A], feed_dict={A_prev: X, K.learning_phase():0})
#     print("out = " + str(out[0][1][1][0]))

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides=(s, s), padding="valid", name=conv_name_base+"2a", kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(F2, (f, f), strides=(1, 1), padding="same", name=conv_name_base+"2b", kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", name=conv_name_base+"2c", kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding="valid", name=conv_name_base+"1", kernel_initializer=glorot_uniform(0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+"1")(X_shortcut)

    X = X + X_shortcut
    X = Activation("relu")(X)

    return X
# tf.reset_default_graph() # TODO 与答案不符
# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder("float", [3, 4, 4, 6])
#     X = np.random.randn(3, 4, 4, 6)
#     A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block="a")
#     test.run(tf.global_variables_initializer())
#     out = test.run([A], feed_dict={A_prev:X, K.learning_phase():0})
#     print("out =" + str(out[0][1][1][0]))
def ResNet50(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name="conv1", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="b")
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="c")

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block="d")

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block="f")

    X = convolutional_block(X, 3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block="c")

    X = AveragePooling2D((2, 2), name="avg_pool")

    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name="fc"+str(classes), kernel_initializer=glorot_uniform(seed=0))

    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model
model = ResNet50(input(64, 64, 3), classes=6)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

model.fit(X_train, Y_train, epochs=2, batch_size=32)
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy =" + str(preds[1]))