'''
'''

import numpy as np
import os
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image

from .data_sample import DataSample


class TXTSample(DataSample):

    def __init__(self, path):
        super().__init__(path)
        
    def get_sample(self):
        path = os.path.join(self.path, [name for name in os.listdir(self.path) if '.txt' in name][0])
        
        with open(path) as file:
            frames = int(file.readline().strip())
            self.sample_id = np.random.randint(frames)

            for i in range(frames):
                if i == self.sample_id:
                    self.sample_data = [float(num) for num in file.readline().strip().split()]
                    break
                else:
                    file.readline()
        
        display('ID:', self.sample_id)
        print(*self.sample_data, sep=', ')
    
    def find_image(self, path):
        self.label = [name for name in os.listdir(path) if '.jpg' in name][self.sample_id]
        path = os.path.join(path, self.label)
        
        self.image = Image.open(path)
        
        self.width = self.image.size[0]
        self.height = self.image.size[1]
    
    def show_image(self):
        title = f'width: {self.width} | height: {self.height}'
        
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle(title)
        
        ax = fig.add_subplot(1, 2, 1)        
        ax.imshow(self.image)
        
        
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        
        i_begin, i_end = 0, 3

        for _ in range(21):
            x, y, z = self.sample_data[i_begin:i_end]
            ax.scatter(x, y, z)
            
            i_begin += 3
            i_end += 3