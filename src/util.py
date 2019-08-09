import numpy as np
from PIL import Image
import os

'''
PIL Image object to np.array
'''
def PIL2array(img):
    return numpy.array(img.getdata(),
                    numpy.uint8).reshape(img.size[1], img.size[0], 3)
