'''
'''

import os
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt


def display_content(module, path, path_images, field_name = None, landmarks = False):
    '''
    '''

    sample = module(path)

    try:
        sample.get_sample()
    except:
        sample.get_sample(field_name)
    
    sample.find_image(path_images)

    if landmarks:
        try:
            sample.show_image(field_name)
        except:
            sample.show_image()
    else:
        sample.show_image()


class DataSample(object):

    def __init__(self, path):
        self.path = path
        self.label = ''
        self.sample_id = ''
        self.sample_data = {}
        self.image = None
    
    def find_image(self, path):
        path = os.path.join(path, self.label)
        
        for img in os.listdir(path):
            if self.sample_id in img:
                self.image = Image.open(os.path.join(path, img))
                break
        
        self.width = self.image.size[0]
        self.height = self.image.size[1]
    
    def show_image(self):
        title = f'width: {self.width} | height: {self.height}'
        
        plt.title(title)        
        plt.imshow(self.image)