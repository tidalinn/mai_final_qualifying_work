'''
'''

import numpy as np
import os
import json
from IPython.display import display
import matplotlib.pyplot as plt

from .data_sample import DataSample


class JSONSample(DataSample):

    def __init__(self, path):
        super().__init__(path)
        file_names = os.listdir(self.path)
        self.sample_id = np.random.randint(len(file_names))
        self.label = file_names[self.sample_id].split('.')[0]
        
    def get_sample(self):
        path = os.path.join(self.path, os.listdir(self.path)[self.sample_id])
        
        with open(path) as file:
            for i, (idx, data) in enumerate(json.load(file).items()):
                if i == self.sample_id:
                    self.sample_id = idx
                    self.sample_data = data
                    break
        
        print('ID:', self.sample_id)
        display(self.sample_data)
    
    def show_image(self, field_name):
        super().show_image()

        for (x, y) in self.sample_data[field_name][0]:
            plt.scatter(x * self.width, y * self.height)