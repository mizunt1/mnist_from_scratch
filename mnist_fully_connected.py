# First task would be to build a network, and pass in one image
# and perform back prop just on that image and change the weights once.
# this will help for back prop https://www.youtube.com/watch?v=tIeHLnjs5U8
# https://www.cl.cam.ac.uk/archive/mjcg/plans/Backpropagation.html
# http://neuralnetworksanddeeplearning.com/chap2.html
# next time, go through Nielsen material above and implement back prop and add in biases

import numpy as np
import mnist_tools as mt

# hyper params
input_len = 784
layer1_len = 100
layer2_len = 100
output_len = 10

pixel_file = "data/trainingSet/0/img_542.jpg"
# pixel intensities of 32*32 as a single vector len is 784
pixels = mt.get_pixels(pixel_file).flatten()

# example hypothetical target
one_hot = np.array([0 for i in range(10)])
one_hot[0] = 1

input_to_h1_weight = mt.weight_2d(input_len, layer1_len)
print("input_to_h1", input_to_h1_weight.shape)
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
    input_norm = input_vector / np.sum(input_vector)
    input_out = mt.relu(input_norm)
    h1_in = np.matmul(input_out, input_to_h1_weight)
    print("h1_in shape", h1_in.shape)
    h1_out = mt.sigmoid(h1_in)
    print("h1 out shape", h1_out.shape)
    h2_in = np.matmul(h1_out, h1_to_h2_weight)
    print("h1 to h2 shape", h1_to_h2_weight.shape)
    print("h2_inshape", h2_in.shape)
    h2_out = mt.softmax(h2_in)
    out_layer_in = np.matmul(h2_out, h2_to_out_weight)
    out_layer_out = mt.loss(target, out_layer_in)
    return input_norm, h1_in, h1_out, h2_in, h2_out, out_layer_in, out_layer_out


def run_back_prop(iterations, input_data, target, starting_weights):
    """
    rememebr that this is still for 1D
    """
    # starting weights dictionary thing:
    input_to_h1_weight = starting_weights['input_to_h1_weight']
    h1_to_h2_weight = starting_weights['h1_to_h2_weight']
    h2_to_out_weight = starting_weights['h2_to_out_weight']
    for i in range(iterations):
        input_layer, h1_in, h1_out, h2_in, h2_out, out_layer_in, out_layer_out = run_model(
            input_data,
            target,
            input_to_h1_weight,
            h1_to_h2_weight,
            h2_to_out_weight)

        new_h2_to_out_weight = mt.hidden_to_final(
            h2_to_out_weight, h2_out, out_layer_out, out_layer_in, target)
        new_h1_to_h2_weight, dEtotal_h2outy = mt.hidden_1_to_hidden_2(
            h1_to_h2_weight, h2_to_out_weight, h1_in, h1_out, h2_in, h2_out, out_layer_in, out_layer_out)
        new_input_to_h1_weight = mt.input_to_hidden_1(
            input_to_h1_weight, h1_to_h2_weight, input_layer, h2_in, h1_in, dEtotal_h2outy)
        print("loss is")
        print(np.average(out_layer_out))
        input_to_h1_weight = new_input_to_h1_weight
        h1_to_h2_weight = new_h1_to_h2_weight
        h2_to_out_weight = new_h2_to_out_weight

run_back_prop(50, pixels, one_hot, starting_weights)
