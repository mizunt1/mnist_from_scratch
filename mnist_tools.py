from PIL import Image
def get_pixels()
    im = Image.open('data/testSample/img_339.jpg')
    pixels = list(im.getdata())
    width, height = im.size
    print(pixels)
# returns 1d pixel, better to return 2d pixel
