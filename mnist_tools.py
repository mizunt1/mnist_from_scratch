from PIL import Image
# could probs do pixel analysis with numpy
import numpy as np


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


def weight_2d(y, x, low=0, high=1):
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

def hidden_to_final(weights, output_vals_h2, output_vals_f, out_layer_in, target_vals, lr=0.01):
    """
    Outputs the optimised weight values for the hidden to final layer of the
    network. The weights should then be replaced by the output of this fn
    Assumed softmax is used in final layer. TODO: Change this to supply options
    for different activation fns
    :params weights: 2D weights matrix of the current weights which are under
     optimisation should be of dimensions (prev number of nodes, current number of nodes)
    :params biases: 1D vector of biases (current number of nodes)
    :params output_vals_h: 1D vector, output of the hidden layer. (prev number of
     nodes)
    :params output_vals_f: 1D vector, output of final layer.
    :params target_vals: vector of target values 1D.
    :params lr: learning rate
    :returns: updated weights which should be used to optimise network
    """

    dEz_dOz = dloss(target_vals, output_vals_f)
    dOoutz_dOinz = dsoftmax(out_layer_in)
    dOinz_dWyz = np.zeros((len(output_vals_h2), len(output_vals_f)))
    for i in range(len(output_vals_h2)):
        for j in range(len(output_vals_f)):
            dOinz_dWyz[j] = output_vals_h2[i]
    # element wise multiplication of first two terms calculated
    print("shape dEz_dOz", dEz_dOz.shape)
    print("shape dOoutz_dOinz", dOoutz_dOinz.shape)
    first_two = np.multiply(dEz_dOz, dOoutz_dOinz)
    first_two_reshaped = np.reshape(first_two, (1,-1))
    print("first_two_reshaped.shape", first_two_reshaped.shape)
    dw = np.multiply(first_two_reshaped, dOinz_dWyz)
    print("dw shape", dw.shape)
    new_weights = weights - lr*dw
    return new_weights


def hidden_1_to_hidden_2(weights, weightyz, input_vals_h1, output_vals_h1, h2_in,
                         output_vals_h2, input_vals_O, output_vals_O, lr=0.01):
    dh2Oy_dh2iny = dsoftmax(h2_in)
    dh2iny_dWxy = np.zeros((len(output_vals_h1), len(output_vals_h2)))
    for i in range(len(output_vals_h1)):
        for j in range(len(output_vals_h2)):
            dh2iny_dWxy[j] = output_vals_h1[i]
    #dEtotal_h2outy vector of len y
    print("output vals O", output_vals_O)
    dE1_dOouty = dsigmoid(output_vals_O)
    dOout1_dOin1 = dsoftmax(input_vals_O)
    mult_with_w = np.multiply(dE1_dOouty, dOout1_dOin1)
    mult_with_w_reshaped = np.reshape(mult_with_w, (1,-1))
    to_be_1d = np.multiply(weightyz, mult_with_w_reshaped)
    dEtotal_h2outy = np.sum(to_be_1d, axis=1)
    dEtotal_h2outy = np.reshape(dEtotal_h2outy, (1, -1))
    first_two_terms = np.multiply(dh2Oy_dh2iny, dEtotal_h2outy)
    dw = np.multiply(dh2iny_dWxy, first_two_terms)
    new_weights = weights - lr*dw
    return new_weights, dEtotal_h2outy
    # element wise multiplication of first two terms calculate

def input_to_hidden_1(weights, weightxy, output_vals_I, input_vals_h2, input_vals_h1,
                      dEtotal_h2outy, lr=0.01):
    # relu activation
    dh1outx_dh1inx = np.ones(len(input_vals_h1))
    dh1outx_dh1inx = np.reshape(dh1outx_dh1inx, (1, -1))
    dh1inx_dWwx = np.reshape(output_vals_I, (-1, 1))
    print("dh1inx_dWwx", dh1inx_dWwx.shape)
    ###### shuoldnt be 3 shuold be len of h1in below ##### use np.tile
    output_vals_I_2d = np.concatenate((dh1inx_dWwx, dh1inx_dWwx, dh1inx_dWwx), axis=1)
    print("output vals 2d", output_vals_I_2d.shape)
    dh1inx_dWwx = np.rollaxis(output_vals_I_2d, 1)
    Wxx = np.zeros(len(weightxy))
    # not entirely sure if this following bit is correct
    for i in range(len(weightxy)):
        Wxx[i] = weightxy[i][i]
    Wxx = np.reshape(Wxx, (-1,1))
    dh2outy_dh2iny = dsigmoid(input_vals_h2)
    dh2outy_dh2iny = np.reshape(dh2outy_dh2iny, (-1, 1))
    dEtotal_dh1outx = np.multiply(dEtotal_h2outy, dh2outy_dh2iny, Wxx)
    first_two = np.multiply(dEtotal_dh1outx, dh1outx_dh1inx)
    first_two = np.reshape(first_two, (-1,1))
    dw = np.multiply(dh1inx_dWwx, first_two)
    new_weights = weights - lr*dw
    return new_weights


if __name__ == "__main__":
    print(get_pixels("data/testSample/img_347.jpg"))
