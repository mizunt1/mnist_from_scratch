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
layer2_len = 100
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

# print("number of images loaded:", len(image_list))
#pixel_file = "data/trainingSet/0/img_542.jpg"
pixel_file = image_list[0]
## print(pixel_file)
# pixel intensities of 32*32 as a single vector len is 784
#pixels = mt.get_pixels(pixel_file).flatten()

# example hypothetical target
one_hot = np.array([0 for i in range(10)])
one_hot[0] = 1

input_to_h1_weight = mt.weight_2d(input_len, layer1_len)
# (784, 100)
h1_to_h2_weight = mt.weight_2d(layer1_len, layer2_len)
# (100, 80)
h2_to_out_weight = mt.weight_2d(layer2_len, output_len)
# (80,10)
starting_weights = {"input_to_h1_weight": input_to_h1_weight,
                    "h1_to_h2_weight": h1_to_h2_weight,
                    "h2_to_out_weight": h2_to_out_weight}

# biases

layer1_bias = mt.bias(layer1_len)
layer2_bias = mt.bias(layer2_len)
output_bias = mt.bias(output_len)

# model


def run_model(input_vector, target, input_to_h1_weight, h1_to_h2_weight,
              h2_to_out_weight):
    input_out = input_vector / np.sum(input_vector)
    # print("input_out", input_out)
    h1_in = np.matmul(input_out, input_to_h1_weight)
    h1_out = mt.relu(h1_in)
    # print("h1_out", h1_out)
    h2_in = np.matmul(h1_out, h1_to_h2_weight)
    # print("h2_in", h2_in)
    h2_out = mt.sigmoid(h2_in)
    # print("h2 out", h2_out)
    out_layer_in = np.matmul(h2_out, h2_to_out_weight)
    # print("out layer in", out_layer_in)
    out_layer_in = out_layer_in /np.sum(out_layer_in)
    out_layer_out = mt.softmax(out_layer_in)
    # print("out_layer_out", out_layer_out)
    out_layer_out_norm = out_layer_out / np.sum(out_layer_out)
    # print("out layer norm", out_layer_out_norm)
    loss_layer_out = mt.loss(target, out_layer_out_norm)
    # # print("loss", loss_layer_out)
    return input_out, h1_in, h1_out, h2_in, h2_out, out_layer_in, loss_layer_out, out_layer_out_norm



def run_back_prop(iterations, data_dict, starting_weights):
    """
    rememebr that this is still for 1D
    """
    batch_size = 10
    num_data_in_each = 60
    # starting weights dictionary thing:
    input_to_h1_weight = starting_weights['input_to_h1_weight']
    h1_to_h2_weight = starting_weights['h1_to_h2_weight']
    h2_to_out_weight = starting_weights['h2_to_out_weight']
    sum_h2_out_weight = np.zeros(h2_to_out_weight.shape)
    sum_h1_h2_weight = np.zeros(h1_to_h2_weight.shape)
    sum_input_h1_weight = np.zeros(input_to_h1_weight.shape)

    for i in range(iterations):
        randint = np.random.randint(0, high=9)
        input_data = data_dict[randint][i//num_data_in_each]
        target = mt.return_one_hot(randint)
        input_layer, h1_in, h1_out, h2_in, h2_out, out_layer_in, loss_layer_out, out_layer_out = run_model(
            input_data,
            target,
            input_to_h1_weight,
            h1_to_h2_weight,
            h2_to_out_weight)

        d_h2_to_out_weight = mt.hidden_to_final(
            h2_to_out_weight, h2_out, out_layer_out, out_layer_in, target)
        d_h1_to_h2_weight, dEtotal_h2outy = mt.hidden_1_to_hidden_2(
            h1_to_h2_weight, h2_to_out_weight, h1_in, h1_out, h2_in, h2_out, out_layer_in, out_layer_out)
        d_input_to_h1_weight = mt.input_to_hidden_1(
            input_to_h1_weight, h1_to_h2_weight, input_layer, h2_in, h1_in, dEtotal_h2outy)

        sum_h2_out_weight = np.add(d_h2_to_out_weight, sum_h2_out_weight)
        sum_h1_h2_weight = np.add(d_h1_to_h2_weight, sum_h1_h2_weight)
        sum_input_h1_weight = np.add(d_input_to_h1_weight, sum_input_h1_weight)

        if i % batch_size == 0:
            print("loss is")
            print(np.average(loss_layer_out))
            print("out layer", out_layer_out)
            print(target)
            print(
                "wrongness (max 1, min 0): ", np.sum(np.absolute(out_layer_out - target))/10)
            # print("out layer in", out_layer_in)
            lr =0.1
            input_to_h1_weight = np.subtract(
                input_to_h1_weight, np.multiply(np.divide(sum_input_h1_weight, batch_size), lr))
            h1_to_h2_weight = np.subtract(
                h1_to_h2_weight, np.multiply(np.divide(sum_h1_h2_weight, batch_size), lr))
            h2_to_out_weight = np.subtract(
                h2_to_out_weight, np.multiply(np.divide(sum_h2_out_weight, batch_size), lr))
            sum_h2_out_weight = np.zeros(h2_to_out_weight.shape)
            sum_h1_h2_weight = np.zeros(h1_to_h2_weight.shape)
            sum_input_h1_weight = np.zeros(input_to_h1_weight.shape)

run_back_prop(1000, data_dict, starting_weights)
