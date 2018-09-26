from PIL import Image
import numpy as np

def get_pixels(file_name):
    im = Image.open(file_name)
    pixels = np.asarray(im.getdata())
    width, height = im.size
    pixels = np.reshape(pixels, (width,height))
    return pixels
# returns 1d pixel, better to return 2d pixel

if __name__=="__main__":
    print(get_pixels("data/testSample/img_347.jpg"))
