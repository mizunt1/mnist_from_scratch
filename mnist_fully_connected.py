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
print("loaded data")

input_to_h1_weight = mt.weight_2d(input_len, layer1_len)
# (784, 100)
h1_to_output_weight = mt.weight_2d(layer1_len, output_len)
# (100, 10)
b1 = mt.bias(layer1_len)
b2 = mt.bias(output_len)

starting_weights = {"input_to_h1_weight": input_to_h1_weight,
                    "h1_to_output_weight": h1_to_output_weight}
starting_bias = {"b1": b1, "b2": b2}


def run_model(input_vector, target, input_to_h1_weight, h1_to_output_weight, b1, b2):
    # normalise input
    # input_out =a1
    input_out = mt.sigmoid(input_vector)

    # (1, 784)
    # might have to do np.add
    # weights = (784, 100)
    # h1_in = z1
    h1_in = np.matmul(input_out, input_to_h1_weight) + b1
    # (100, 1)
    # h1_out = a2
    h1_out = mt.sigmoid(h1_in)
    # (100,1)
    # out_in = z2
    out_in = np.matmul(h1_out, h1_to_output_weight) + b2
    # weights = (100, 10)
    # (100,1)
    # out_out = a3
    out_out = mt.sigmoid(out_in)
    # (10, 1)
    loss = mt.cost(target, out_out)
    return input_out, h1_in, h1_out, out_in, out_out, loss


def run_back_prop(iterations, data_dict, starting_weights, starting_bias):
    """
    rememebr that this is still for 1D
    """
    batch_size = 10
    num_data_in_each = 60
    # starting weights dictionary thing:
    input_to_h1_weight = starting_weights['input_to_h1_weight']
    h1_to_output_weight = starting_weights['h1_to_output_weight']
    b1 = starting_bias['b1']
    b2 = starting_bias['b2']

    sum_input_h1_weight = np.zeros(input_to_h1_weight.shape)
    sum_h1_out_weight = np.zeros(h1_to_output_weight.shape)
    sum_b1 = np.zeros(b1.shape)
    sum_b2 = np.zeros(b2.shape)

    for i in range(iterations):
        # initialise data to input in to model
        randint = np.random.randint(0, high=9)
        input_data = data_dict[randint][i//num_data_in_each]
        target = mt.return_one_hot(randint)
        input_out, h1_in, h1_out, out_in, out_out, loss = run_model(
            input_data,
            target,
            input_to_h1_weight,
            h1_to_output_weight, b1, b2)
        # compute changes to weights and sum them
        dw_2, d_b2 = mt.dfinal(target, out_out, out_in, h1_in)
        dw_1, d_b1 = mt.dw1(h1_to_output_weight, d_b2, h1_in, input_out)
        # dw_2 is derivative of loss function
        # sum changes
        sum_input_h1_weight = np.add(dw_1, sum_input_h1_weight)
        sum_h1_output_weight = np.add(dw_2, sum_h1_out_weight)
        sum_b1 = np.add(d_b1, sum_b1)
        sum_b2 = np.add(d_b2, sum_b2)

        if i % batch_size == 0:
            print("loss is")
            print(np.average(loss_layer_out))
            print("out layer", out_layer_out)
            print(target)
            print(
                "wrongness (max 1, min 0): ",
                np.sum(np.absolute(out_layer_out - target))/10)
            # print("out layer in", out_layer_in)
            lr =0.1
            input_to_h1_weight = np.subtract(
                input_to_h1_weight, np.multiply(
                    np.divide(sum_input_h1_weight, batch_size), lr))
            h1_to_output_weight = np.subtract(
                h1_to_ouput_weight, np.multiply(
                    np.divide(sum_h1_output_weight, batch_size), lr))

            b1 = np.subtract(b1, np.multiply(np.divide(sum_b1, batch_size), lr))
            b2 = np.subtract(b2, np.multiply(np.divide(sum_b2, batch_size), lr))


            sum_input_h1_weight = np.zeros(input_to_h1_weight.shape)
            sum_h1_out_weight = np.zeros(h1_to_out_weight.shape)
            sum_b1 = np.zeros(b1.shape)
            sum_b1 = np.zeros(b2.shape)

run_back_prop(1000, data_dict, starting_weights, starting_bias)
