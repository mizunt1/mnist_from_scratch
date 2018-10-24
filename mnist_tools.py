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
    y = (1+np.exp(-x))**-1
    return y


def dsigmoid(x):
    y = sigmoid(x)*(1-sigmoid(x))
    return y


def ce_loss(output, target):
    loss = (np.log(output) + (1-target) * np.log(1 - output))*-1
    return loss


def d_ce_loss(output, target):
    d = -1*((target * (1/output)) + ((1 - target) * (1/(1 - output))))
    return d

def softmax(vector):
    """
    Input a vector of dimension n, output vector of dimension n
    after applying the softmax fn.
    :params vector: n dimensional np.array vector
    """
    output = np.zeors(len(vector))
    denom = 0
    for i in range(len(vector)):
        denom += np.exp(a[i])
    for i in range(len(a)):
        output[i] = np.exp(a[i])
    return output

def dsoftmax(vector):
    """
    value of differential of softmax at those vectors
    """
    output = np.zeros(len(vector))
    for i in range(len(vectors)):
        denom += np.exp()
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
    loss = np.zeros(len(input)):
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
    loss = np.zeros(len(input)):
    for i in range(len(input)):
        loss[i] = -1*(target[i]*(1/input[i]) + (1-target[i])*(1/(1-input[i])))
    return loss

def hidden_to_final(weights, biases, output_vals_h, output_vals_f, target_vals):
    """
    Outputs the optimised weight values for the hidden to final layer of the
    network. The weights should then be replaced by the output of this fn
    :params weights: 2D weights matrix of the current weights which are under 
     optimisation should be of dimensions (prev number of nodes, current number of nodes)
    :params biases: 1D vector of biases (current number of nodes)
    :params output_vals_h: 1D vector, output of the hidden layer. (prev number of
     nodes)
    :params output_vals_f: 1D vector, output of final layer.
    :params target_vals: vector of target values 1D.
    :returns: updated weights which should be used to optimise network
    """
    
    dEz_dOz = dloss(target,output_vals_f)
    dOoutz_dOinz = dsoftmax(output_vals_h)
    dOinz_Wyz = np.zeros((len(output_vals_h), len(output_vals_f)))
    for i in range(len(output_vals_h)):
        for j in range(len(output_vals_f)):
            dOinz_Wyz[j] = output_vals_h[i]
    # element wise multiplication of first two terms calculated
    first_two = np.multiply(dEz_dOz, dOoutz_dOinz)
    first_two_reshaped = np.reshape(first_two, (-1,1))
    output = np.multiply(first_two_reshaped, dOinz_Wyz)
    

    

if __name__ == "__main__":
    print(get_pixels("data/testSample/img_347.jpg"))
