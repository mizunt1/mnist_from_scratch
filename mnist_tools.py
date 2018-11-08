from PIL import Image
# could probs do pixel analysis with numpy
import numpy as np
lr = 0.1

def loss2(target, pred):
    init = np.zeros(target.shape)
    i = np.argmax(target)
    pred_idx = pred[i]
    val = np.log(pred_idx)
    init[i] = val
    return init

def return_one_hot(integer):
    blank = np.zeros(10)
    blank[integer] = 1
    return blank


def get_pixels(file_name):
    """
    takes a file path to an image and returns an array of pixels
    """
    im = Image.open(file_name)
    pixels = np.asarray(im.getdata())
    width, height = im.size
    pixels = np.reshape(pixels, (width, height))
    return pixels


def relu(x):
    output = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= 0:
            output[i] = 0
        else:
            output[i] = x[i]
    return output


def weight_2d(y, x, low=-1, high=1):
    """
    Create a 2d matrix with columns x and rows y filled with random
    numbers from -1 to 1
    """
    matrix = np.random.uniform(low=low, high=high, size=(y, x))
    return matrix


def bias(y, low=-1, high=1):
    """
    creates a randomised bias vector
    """
    vector = np.random.uniform(low=low, high=high, size=(y))
    return vector


def sigmoid(x):
    output = np.zeros(len(x))
    for i in range(len(x)):
        output[i] = (1+np.exp(-x[i]))**-1
    return output


def sigmoid_single(x):
    out = (1+np.exp(-x))**-1
    return out


def dsigmoid(x):
    output = np.zeros(len(x))
    for i in range(len(x)):
        output[i] = sigmoid_single(x[i])*(1-sigmoid_single(x[i]))
    return output


def softmax(vector):
    """
    Input a vector of dimension n, output vector of dimension n
    after applying the softmax fn.
    :params vector: n dimensional np.array vector
    """
    output = np.zeros(len(vector))
    denom = 0
    for i in range(len(vector)):
        denom += np.exp(vector[i])
    for i in range(len(vector)):
        output[i] = np.exp(vector[i])

    return output / denom


def dsoftmax(vector):
    """
    value of differential of softmax at those vectors
    """
    output = np.zeros(len(vector))
    denom = 0
    for i in range(len(vector)):
        denom += np.exp(vector[i])
    for i in range(len(vector)):
        num = (vector[i] * (np.sum(vector) - vector[i]))/denom
        output[i] = num
    return output


def loss(target, input):
    """
    input target vector and input vector and calculate the cross entropy loss
    :params target: target vector n dimensional vector of targets
    :params input: input vector of n dimensional vector of inputs
    :returns: n dimensional vector of loss
    """
    loss = np.zeros(len(input))
    for i in range(len(input)):
        loss[i] = -1*(target[i]*np.log(input[i]) + (1-target[i])*np.log(1-input[i]))
    return loss


def dloss(target, input):
    """
    calculates differential of cross entropy loss at those values in the input vector
    :params target: target vector, n dimensional vector
    :params input: target vector, n dimensional vector
    :returns: An n dimensional vector
    """
    loss = np.zeros(len(input))
    for i in range(len(input)):
        loss[i] = -1*(target[i]*(1/input[i]) + (1-target[i])*(1/(1-input[i])))
    return loss

def dloss2(target, pred):
    init = np.zeros(target.shape)
    idx = np.argmax(target)
    val=1/pred[idx]
    init[idx] = val
    return init
    
def hidden_to_output(target, out_out, out_in, loss, h1_out):
    # dloss_dout_out
    loss_init = np.zeros(len(loss))
    idx = np.argmax(target)
    loss_init[idx] = 1/out_out[idx]
    dloss_dout_out = loss_init
    # dout_out_d_out_in
    dout_out_d_out_in = dsoftmax(out_in)
    # d_out_in_dW
    h1_out = np.reshape(h1_out,(-1,1))
    d_out_in_dw = np.tile(h1_out, len(out_out))
    #multiply all three
    dw = dloss_dout_out*dout_out_d_out_in*d_out_in_dw
    for_bias2 =  dloss_dout_out*dout_out_d_out_in
    return dw, for_bias2

def db2(for_bias2):
    return for_bias2

def input_to_hidden(dw1, h1_in, input_vals):
    dw1 = np.rollaxis(dw1, axis=1)
    # dh1_out_dh1_in
    print(dw1.shape)
    dh1_out_dh1_in = dsigmoid(h1_in)
    print("dh1_out_dh1_in",dh1_out_dh1_in.shape)
    # dh1_in_dw1
    dh1_out_dh1_in = np.reshape(dh1_out_dh1_in,(1,-1))
    input_vals = np.reshape(input_vals, (-1,1))
    dh1_in_dw1 = np.tile(input_vals, len(h1_in))
    print("dh1_in_dw1", dh1_in_dw1.shape)
    dw = dw1*dh1_in_dw1*dh1_out_dh1_in
    for_bias = dh1_out_dh1_in
    return dw, for_bias

def db1(dw2, dw1):
    return dw1*dw2

