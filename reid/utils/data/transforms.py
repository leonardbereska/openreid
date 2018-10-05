from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math


class RectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        h_ = self.height
        w_ = self.width
        if h == h_ and w == w_:
            return img
        c = (w/2., h/2.)

        img = img.crop((c[0]-w_/2, c[1]-h_/2, c[0]+w_/2, c[1]+h_/2))
        # from matplotlib import pyplot as plt
        # plt.imshow(img)
        # plt.show()
        return img.resize((self.width, self.height), self.interpolation)


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, crop_scale=0.64, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.min_area = crop_scale
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                # from matplotlib import pyplot as plt
                # plt.imshow(img_)
                # plt.show()
                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)
