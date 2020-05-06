import random
from bokeh.plotting import figure
from bokeh.embed import components
from imageio import imread
from numpy import pi
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import matplotlib.pyplot as plt




ALLOWED_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])
LETTER_SET = list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
IMAGE_LABELS = ['Headphone', 'Mouse', 'Camera', 'Smartphone',
                'Glasses', 'Shoes', 'Watch', 'Laptop']


def is_allowed_file(filename):
    """ Checks if a filename's extension is acceptable """
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext


def generate_random_name(filename):
    """ Generate a random name for an uploaded file. """
    ext = filename.split('.')[-1]
    rns = [random.randint(0, len(LETTER_SET) - 1) for _ in range(3)]
    chars = ''.join([LETTER_SET[rn] for rn in rns])

    new_name = "{new_fn}.{ext}".format(new_fn=chars, ext=ext)
    new_name = secure_filename(new_name)

    return new_name

def make_thumbnail(filepath):
    """ Converts input image to 224px by 224px thumbnail if not that size
    and save it back to the source file """
    img = Image.open(filepath)
    thumb = None
    w, h = img.size

    # if it is exactly 224x224, do nothing
    if w == 224 and h == 224:
        return True

    # if the width and height are equal, scale down
    if w == h:
        thumb = img.resize((224, 224), Image.BICUBIC)
        thumb.save(filepath)
        return True

    # when the image's width is smaller than the height
    if w < h:
        # scale so that the width is 224px
        ratio = w / 224.
        w_new, h_new = 224, int(h / ratio)
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        top, bottom = 0, 0
        margin = h_new - 224
        top, bottom = margin // 2, 224 + margin // 2
        box = (0, top, 224, bottom)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True

    # when the image's height is smaller than the width
    if h < w:
        # scale so that the height is 224px
        ratio = h / 224.
        w_new, h_new = int(w / ratio), 224
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        left, right = 0, 0
        margin = w_new - 224
        left, right = margin // 2, 224 + margin // 2
        box = (left, 0, right, 224)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True
    return False

