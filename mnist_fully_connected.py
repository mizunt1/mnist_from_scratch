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
layer2_len = 80
output_len = 10

pixel_file = "data/trainingSample/0/img_542.jpg"
# pixel intensities of 32*32 as a single vector len is 784
pixels = mt.get_pixels(pixel_file).flatten()

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
    input_layer_out = mt.relu(np.matmul(input_vector, input_to_h1_weight))
    h1_out = mt.relu(np.matmul(input_layer_out, h1_to_h2_weight))
    h2_out = mt.softmax(np.matmul(h1_out, h2_to_out_weight))
    final_out = mt.loss(target, h2_out)
    return final_out, h2_out, h1_out, input_layer_out


def run_back_prop(iterations, input_data, target, starting_weights):
    """
    rememebr that this is still for 1D
    """
    # starting weights dictionary thing:
    input_to_h1_weight = starting_weights['input_to_h1_weight']
    h1_to_h2_weight = starting_weights['new_h1_to_h2_weight']
    h2_to_out_weight = starting_weights['new_h2_to_out_weight']
    for i in range(iterations):
        if i != 0:
            input_to_h1_weight = new_input_to_h1_weight
            h1_to_h2_weight = new_h1_to_h2_weight
            h2_to_out_weight = new_h2_to_out_weight
        final_out, h2_out, h1_out, input_layer_out = run_model(input_data,
                                                               target,
                                                               input_to_h1_weight,
                                                               h1_to_h2_weight,
                                                               h2_to_out_weight)
        new_h2_f_weights = mt.hidden_to_final(
            h2_to_out_weight, h2_out, final_out, target)
        new_h1_h2_weights = mt.hidden_1_to_hidden_2(h2)

loss = mt.loss(one_hot, final_out)

loss_ave = np.average(loss)
print(loss)
print(loss_ave)
