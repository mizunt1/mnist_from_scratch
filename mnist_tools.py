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
    blank = np.zeros((10, 1))
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
    vector = np.random.uniform(low=low, high=high, size=(y, 1))
    return vector


def sigmoid(x):
    output = np.zeros((len(x), 1))
    for i in range(len(x)):
        output[i] = (1+np.exp(-x[i]))**-1
    return output


def relu(x):
    output = np.zeros((len(x), 1))
    for i in range(len(x)):
        if x[i] <= 0:
            output[i] = 0
        else:
            output[i] = x[i]
    return output


def sigmoid_single(x):
    out = (1+np.exp(-x))**-1
    return out


def dsigmoid(x):
    # print("sinput sigmoid", x.shape)
    output = np.zeros((len(x), 1))
    for i in range(len(x)):
        output[i] = sigmoid_single(x[i])*(1-sigmoid_single(x[i]))
    # print("ouput", output.shape)
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
    val=1 / pred[idx]
    init[idx] = val
    return init


def cost(target, pred):
    out = (pred-target)**2
    num_items = len(target)
    out = sum(out) / (2*num_items)
    return out


def dw1(w_t2, dz_t2, z_t1, a_t0):
    """
    w_t2: weights of layer above the layer on interest
    dz_t2: dC/dz of layer above tha layer on interest
    z_t1: dC/dz of layer of interst
    a_t0: a of layer below the layer of interest
    """
    w_t2_trans = np.transpose(w_t2)
    # print("w_t2_trans should be (100,10)", w_t2_trans.shape)
    # print("dz_t2 should be (100,1)", dz_t2.shape )
    step_one = np.dot(w_t2_trans, dz_t2)
    step_one = np.reshape(step_one, (-1, 1))
    # print("step one shape shuold be (100,1)", step_one.shape)
    dsigmoid_z = dsigmoid(z_t1)
    # print("dsigmoid shape should be (100, 1)", dsigmoid_z.shape)
    dz_t1 = np.multiply(step_one, dsigmoid_z)
    # print("dz_t1 should be (100,1)", dz_t1.shape)
    # print("a_t0 should be (784,1)", a_t0.shape)
    a_t0_sq = np.squeeze(a_t0)
    dz_t1_sq = np.squeeze(dz_t1)
    # print("a_t0_sq shape should be (784,)", a_t0_sq.shape)
    # print("dz_t1_sq shape (100,)", dz_t1_sq.shape)
    frame = np.tile(a_t0_sq, (len(dz_t1_sq),1))
    for i in range(len(dz_t1_sq)):
        for j in range(len(a_t0)):
            frame[i][j] = a_t0[j][0] * dz_t1[i][0]
    dw_t1 = frame

    # note dz_t1 is the same as db_t1 bias change for layer of interest
    return dw_t1, dz_t1


def dfinal(target, pred, z2, a1):
    # print("target", target.shape)
    # print("pred", pred.shape)
    diff = (pred - target)
    # print("diff shuold be (10,1)", diff.shape)
    dz = np.multiply(diff, dsigmoid(z2))
    # print("dsignmoid should be (10,1)", dz.shape)
    # print("dz shape should be (10,1)")
    a1_sq = np.squeeze(a1)
    dz_sq = np.squeeze(dz)
    frame = np.tile(a1_sq, (len(dz_sq),1))
    # print("a1_sq", a1_sq.shape)
    # print("dz_sq", dz_sq.shape)
    # print("frame shape should be (10, 100)", frame.shape)
    for i in range(len(dz_sq)):
        for j in range(len(a1_sq)):
            frame[i][j] = a1_sq[j] * dz_sq[i]
    dw = frame
    # print("dz out", dz.shape)
    return dw, dz
