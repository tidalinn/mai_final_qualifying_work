'''
'''

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def get_random_image(path: str) -> str:
    random_index = np.random.randint(len(os.listdir(path)))
    random_name = os.listdir(path)[random_index]
    image_path = f'{path}/{random_name}'
    return image_path


def get_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_image(path: str, random: bool = False) -> None:
    if random:
        image = get_image(get_random_image(path))
    else:
        image = get_image(path)
    
    plt.figure(figsize=(5, 5))
    plt.title(f'Image shape: {image.shape}')

    plt.imshow(image)
    
    plt.axis('off')
    plt.show()