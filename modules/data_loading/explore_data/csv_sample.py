'''
'''

import pandas as pd
from IPython.display import display

from .data_sample import DataSample


class CSVSample(DataSample):
    
    def __init__(self, path):
        super().__init__(path)
        
    def get_sample(self, field_name):
        data = pd.read_csv(self.path).iloc[0]

        self.sample_id = data[field_name]
        self.sample_data = dict(data.drop(field_name))

        print('ID:', self.sample_id)
        display(self.sample_data)