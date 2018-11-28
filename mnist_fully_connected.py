# First task would be to build a network, and pass in one image
# and perform back prop just on that image and change the weights once.
# this will help for back prop https://www.youtube.com/watch?v=tIeHLnjs5U8
# https://www.cl.cam.ac.uk/archive/mjcg/plans/Backpropagation.html
# http://neuralnetworksanddeeplearning.com/chap2.html
# next time, go through Nielsen material above and implement back prop and add in biases

import numpy as np
import os
from PIL import Image

import mnist_tools as mt
# hyper params
input_len = 784
layer1_len = 100
output_len = 10

import glob
data_dict = {}
for i in range(10):
    image_list = []
    file_path = os.path.join('data', 'trainingSet', str(i), '*.jpg')
    for filename in glob.glob(file_path):
        im = Image.open(filename)
        pixels = np.asarray(im.getdata())
        width, height = im.size
        pixels = np.reshape(pixels, (width, height))
        pixels = pixels.flatten()
        image_list.append(pixels)
    data_dict[i] = image_list
# print("loaded data")

W1 = mt.weight_2d(layer1_len, input_len)
# we want this to be (100, 784)
# (784, 100)
W2 = mt.weight_2d(output_len, layer1_len)
# we want this to be (100, 10)
# (100, 10)
b1 = mt.bias(layer1_len)
# we want this to be (100,1) OR (100,) not sure
b2 = mt.bias(output_len)
# we want this to be (10,1) or (10,) not sure

starting_weights = {"W1": W1,
                    "W2": W2}
starting_bias = {"b1": b1, "b2": b2}


def run_model(input_vector, target, W1, W2, b1, b2):
    # normalise input
    # a0 =a0
    a0 = mt.sigmoid(input_vector)
    # want this to be (784,1)
    # print("run mdel w2", W2.shape)
    # weights = (784, 100)
    # z1 = z1

    b1 = np.reshape(b1, (-1, 1))

    a0 = np.reshape(a0, (-1, 1))

    # print("a0 shape (784,1)", a0.shape)

    z1 = np.matmul(W1, a0) + b1
    # we want z1 to be (z1 = (100,1))
    # print("h1 in shape (100,1)", z1.shape)
    # (100, 1)
    # a1 = a1
    a1 = mt.sigmoid(z1)
    a1 = np.reshape(a1, (-1, 1))
    # we want this to be h1 out = a1 = (100,1)
    # print("a1 (100, 1)", a1.shape)
    # matmul goes wrong here
    # print("w2 shape (10,100)", W2.shape)
    # print("b2 (10,1)", b2.shape)
    b2 = np.reshape(b2, (-1, 1))
    #### TODO: sort out shapes
    # z2 = np.matmul(W2, a1) + b2

    z2 = np.matmul(W2, a1) + b2
    # W2 = w2 we want this to be (10, 100)
    # z2 = z2 we want this to be = (10, 1)
    # print("z2 (10, 1)", z2.shape)
    # weights = (100, 10)
    # (100,1)
    # a2 = a2
    a2 = mt.sigmoid(z2)
    # (10, 1)
    # print("a2 (10,1)", a2.shape)
    loss = mt.cost(target, a2)
    return a0, z1, a1, z2, a2, loss


def run_back_prop(iterations, data_dict, starting_weights, starting_bias):
    """
    rememebr that this is still for 1D
    """
    batch_size = 1
    num_data_in_each = 60
    # starting weights dictionary thing:
    W1 = starting_weights['W1']
    W2 = starting_weights['W2']

    b1 = starting_bias['b1']
    b2 = starting_bias['b2']

    sum_w1 = np.zeros(W1.shape)
    sum_w2 = np.zeros(W2.shape)
    sum_b1 = np.zeros(b1.shape)
    sum_b2 = np.zeros(b2.shape)
    # print("sum sum", sum_b1.shape, sum_b2.shape)
    for i in range(iterations):
        # initialise data to input in to model
        randint = np.random.randint(0, high=9)
        input_data = data_dict[randint][i//num_data_in_each]
        target = mt.return_one_hot(randint)
        a0, z1, a1, z2, a2, loss = run_model(
            input_data,
            target,
            W1,
            W2, b1, b2)
        # compute changes to weights and sum them
        # print("a2 shape should be (10,1)", a2.shape)
        # print("z2 shape should be (10,1)", z2.shape)
        # print("z1 shape should be (100,1)", z1.shape)
        dw_2, dz_t2 = mt.dfinal(target, a2, z2, a1)
        d_b2 = dz_t2
        # print("db2 out", d_b2.shape)
        # print("db2 out2", d_b2.shape)
        dw_1, dz_t1 = mt.dw1(W2, dz_t2, z1, a0)
        d_b1 = dz_t1
        # print("d_b1 out", d_b1.shape)
        # dw_2 is derivative of loss function
        # sum changes
        sum_w1 = np.add(dw_1, sum_w1)
        sum_w2 = np.add(dw_2, sum_w2)
        sum_b1 = np.add(d_b1, sum_b1)
        sum_b2 = np.add(d_b2, sum_b2)
        if i % batch_size == 0:
            print("loss is")
            print(np.average(loss))
            print("out layer")
            print(a2)
            print(target)
            print(
                "wrongness (max 1, min 0): ",
                np.sum(np.absolute(a2 - target))/10)
            lr = 0.1
            W1 = np.subtract(
                W1, np.multiply(
                    np.divide(sum_w1, batch_size), lr))
            W2 = np.subtract(
                W2, np.multiply(
                    np.divide(sum_w2, batch_size), lr))
            # print(sum_b1.shape, sum_b2.shape)
            b1 = np.subtract(b1, np.multiply(np.divide(sum_b1, batch_size), lr))
            b2 = np.subtract(b2, np.multiply(np.divide(sum_b2, batch_size), lr))
            # print("cc", b2.shape)
            sum_w1 = np.zeros(W1.shape)
            sum_w2 = np.zeros(W2.shape)
            sum_b1 = np.zeros(b1.shape)
            sum_b2 = np.zeros(b2.shape)

run_back_prop(1000, data_dict, starting_weights, starting_bias)
