from PIL import Image
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



if __name__=="__main__":
    print(get_pixels("data/testSample/img_347.jpg"))
