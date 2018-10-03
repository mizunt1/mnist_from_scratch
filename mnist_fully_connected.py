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

pixel_file = "data/trainingSet/0/img_41637.jpg"
pixels = mt.get_pixels(pixel_file).flatten()
sigmoid_vec = np.vectorize(mt.sigmoid)
one_hot = np.array([0 for i in range(10)])
one_hot[0] = 1

layer1_weight = mt.weight_2d(input_len, layer1_len)
layer2_weight = mt.weight_2d(layer1_len, layer2_len)
output_weight = mt.weight_2d(layer2_len, output_len)
# must add biases

# model

layer1_out = sigmoid_vec(np.matmul(pixels, layer1_weight))
layer2_out = sigmoid_vec(np.matmul(layer1_out, layer2_weight))
final_out = sigmoid_vec(np.matmul(layer2_out, output_weight))

print(final_out)
print(one_hot)
loss = (final_out - one_hot)**2
print(loss)
