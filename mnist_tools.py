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


if __name__ == "__main__":
    print(get_pixels("data/testSample/img_347.jpg"))
