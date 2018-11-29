# First task would be to build a network, and pass in one image
# and perform back prop just on that image and change the weights once.
# this will help for back prop https://www.youtube.com/watch?v=tIeHLnjs5U8
# https://www.cl.cam.ac.uk/archive/mjcg/plans/Backpropagation.html
# http://neuralnetworksanddeeplearning.com/chap2.html
# next time, go through Nielsen material above and implement back prop and add in biases

import numpy as np
import os
from PIL import Image
import h5py
import mnist_tools as mt
# hyper params
input_len = 784
layer1_len = 100
output_len = 10

import os

import glob


def create_datadict(file_path):
    data_dict = {}
    for i in range(10):
        image_list = []
        file_path_is = os.path.join(file_path, str(i), '*.jpg')
        for filename in glob.glob(file_path_is):
            im = Image.open(filename)
            pixels = np.asarray(im.getdata())
            width, height = im.size
            pixels = np.reshape(pixels, (width, height))
            pixels = pixels.flatten()
            image_list.append(pixels)
        data_dict[i] = image_list
    return data_dict




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


def checkpoint(w1, w2, b1, b2, filename):
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('w1', data=w1)
    h5.create_dataset('w2', data=w2)
    h5.create_dataset('b1', data=b1)
    h5.create_dataset('b2', data=b2)
    h5.close()


def run_model(input_vector, target, W1, W2, b1, b2):
    # normalise input
    # a0 =a0

    # input_vector = input_vector / np.sum(input_vector)
    # a0 = mt.relu(input_vector)
    a0 = mt.sigmoid(input_vector)
    # a0 = input_vector
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
    # a1 = mt.relu(z1)
    a1 = mt.sigmoid(z1)
    a1 = np.reshape(a1, (-1, 1))
    # we want this to be h1 out = a1 = (100,1)
    # print("a1 (100, 1)", a1.shape)
    # matmul goes wrong here
    # print("w2 shape (10,100)", W2.shape)
    # print("b2 (10,1)", b2.shape)
    b2 = np.reshape(b2, (-1, 1))
    # z2 = np.matmul(W2, a1) + b2

    z2 = np.matmul(W2, a1) + b2
    # W2 = w2 we want this to be (10, 100)
    # z2 = z2 we want this to be = (10, 1)
    # print("z2 (10, 1)", z2.shape)
    # weights = (100, 10)
    # (100,1)
    # a2 = a2
    # a2 = mt.relu(z2)
    a2 = mt.sigmoid(z2)
    # (10, 1)
    # print("a2 (10,1)", a2.shape)
    loss = mt.cost(target, a2)
    return a0, z1, a1, z2, a2, loss


def check_testset(W1, W2, b1, b2, file_path="data/testSet", num_files=5):
    num_correct = 0
    total = 0
    data_dict = create_datadict(file_path)
    for i in range(40):
        randint = np.random.randint(0, high=9)
        randchoose = np.random.randint(0, high=num_files)
        input_data = data_dict[randint][randchoose]
        target = mt.return_one_hot(randint)
        a0, z1, a1, z2, a2, loss = run_model(
            input_data,
            target,
            W1,
            W2, b1, b2)
        a2 = np.argmax(np.ndarray.flatten(a2))
        if randint == a2:
            num_correct += 1
        total += 1
    return num_correct, total


def run_back_prop(iterations, data_dict, starting_weights, starting_bias,
                  checkpoint_load=None, checkpoint_save=None, save_interval=1000, num_files=3000):
    """
    rememebr that this is still for 1D
    """
    directory = checkpoint_save
    if checkpoint_save is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
    batch_size = 1
    # starting weights dictionary thing:
    W1 = starting_weights['W1']
    W2 = starting_weights['W2']

    b1 = starting_bias['b1']
    b2 = starting_bias['b2']
    if checkpoint_load is not None:
        hf = h5py.File(checkpoint_load, 'r')
        W1 = hf['w1']
        W2 = hf['w2']
        b1 = hf['b1']
        b2 = hf['b2']
    sum_w1 = np.zeros(W1.shape)
    sum_w2 = np.zeros(W2.shape)
    sum_b1 = np.zeros(b1.shape)
    sum_b2 = np.zeros(b2.shape)
    # print("sum sum", sum_b1.shape, sum_b2.shape)

    total_correct = 0
    check = 0
    for i in range(iterations):
        # initialise data to input in to model
        randint = np.random.randint(0, high=9)
        randchoose = np.random.randint(0, high=num_files)
        input_data = data_dict[randint][randchoose]
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
            print(np.average(loss))
            a2 = np.ndarray.flatten(a2)
            a2_pred = np.argmax(a2)
            print("out layer argmax", a2_pred)
            target = np.ndarray.flatten(target)
            target = np.argmax(target)
            print("target argmax", target)
            if target == a2_pred:
                total_correct += 1
                print("correct!!")
            check += 1
            total = total_correct / (check)
            print("correctness training set (cumulative): max 1, min 0", total)
            if i % 200 == 0:
                num_correct_test, total_test = check_testset(W1, W2, b1, b2)
                print("~~~~correctness test set: max 1, min 0", num_correct_test/total_test)
                print("correct guesses: ", num_correct_test, "out of: ", total_test)
            if i % save_interval == 0 and (checkpoint_save is not None):
                print("check pointing")
                filename = os.path.join(checkpoint_save, "checkpoints" + str(i) + ".h5")
                checkpoint(W1, W2, b1, b2, filename)
            lr = 0.1
            W1 = W1 - ((sum_w1 / batch_size) * lr)
            W2 = W2 - ((sum_w2 / batch_size) * lr)
            # print(sum_b1.shape, sum_b2.shape)
            b1 = b1 - (sum_b1 / batch_size) * lr
            b2 = b2 - (sum_b2 / batch_size) * lr
            # print("cc", b2.shape)
            sum_w1 = np.zeros(W1.shape)
            sum_w2 = np.zeros(W2.shape)
            sum_b1 = np.zeros(b1.shape)
            sum_b2 = np.zeros(b2.shape)
    num_correct_test, total_test = check_testset(W1, W2, b1, b2)
    print("~~~~correctness test set: max 1, min 0", num_correct_test/total_test)
    print("correct guesses: ", num_correct_test, "out of: ", total_test)
    hf.close()
data_dict = create_datadict('data/trainingSet')

run_back_prop(10000, data_dict, starting_weights, starting_bias, checkpoint_save="check6", checkpoint_load='check5/checkpoints9000.h5')

# run_back_prop(10000, data_dict, starting_weights, starting_bias, checkpoint_save=True)
