'''
'''

import os
from PIL import Image
import matplotlib.pyplot as plt


def display_image(path, depth_map = False):
    img = ImageViewer(path)
    
    if depth_map:
        img.show_depth_image()
    else:
        img.show_image()


class ImageViewer():
    
    def __init__(self, path):
        self.path = os.path.join(path, os.listdir(path)[0])
        self.image = Image.open(self.path)
        self.width = self.image.size[0]
        self.height = self.image.size[1]
        self.title = f'width: {self.width} | height: {self.height}'
    
    def show_image(self):
        plt.title(self.title)
        plt.imshow(self.image)

    def show_depth_image(self):
        image_gray = self.image.convert('P')

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        plt.suptitle(self.title)

        axes[0].imshow(image_gray)
        axes[0].set_title('init')

        axes[1].imshow(self.image)
        axes[1].set_title('depth')

        plt.show()