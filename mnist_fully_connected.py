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
layer1_len = 400
layer2_len = 100
output_len = 10

pixel_file = "data/trainingSample/0/img_542.jpg"
pixels = mt.get_pixels(pixel_file).flatten()

# example hypothetical target
one_hot = np.array([0 for i in range(10)])
one_hot[0] = 1

layer1_weight = mt.weight_2d(input_len, layer1_len)
# (784, 400)
layer2_weight = mt.weight_2d(layer1_len, layer2_len)
# (400, 100)
output_weight = mt.weight_2d(layer2_len, output_len)
# (100,10)

# biases
layer1_bias = mt.bias(layer1_len)
layer2_bias = mt.bias(layer2_len)
output_bias = mt.bias(output_len)

# model
layer1_out = mt.sigmoid(np.matmul(pixels, layer1_weight))
layer2_out = mt.sigmoid(np.matmul(layer1_out, layer2_weight))
final_out = mt.sigmoid(np.matmul(layer2_out, output_weight))


loss = mt.loss(one_hot, final_out)

loss_ave = np.average(loss)
print(loss)
print(loss_ave)

