'''
'''

import pandas as pd
import os
from pathlib import WindowsPath


def list_folder(path: WindowsPath):
    print(f'{path.name} dataset:', os.listdir(path))


def concat_images_labels(path: WindowsPath) -> None:
    pd.DataFrame(
        zip(
            os.listdir(path / 'images'), 
            os.listdir(path / 'labels')
        )
    ).to_csv(path / f"{path.name}.csv", index=False, header=False)