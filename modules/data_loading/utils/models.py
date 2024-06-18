'''
'''

import os
import requests


def load_yolo_weights(model_name: str, url: str):
    if 'weights' not in os.listdir():
        os.mkdir('weights')
    
    with open(f'weights/{model_name}', 'wb') as file:
        file.write(requests.get(url).content)
    
    print('\nSaved weights:', model_name)