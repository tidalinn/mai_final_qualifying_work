'''
'''

import pandas as pd
import os
from IPython.display import display


def display_folder_content(path, inner_dir_name = None, return_data = False):
    '''
    '''
    
    folder = WalkThroughDir(path)
    folder.get_dir_content()

    display(folder.data)

    if inner_dir_name is not None:
        folder.count_classes(inner_dir_name)
    
    if return_data:
        return folder
    

class WalkThroughDir():
    
    def __init__(self, path):
        self.path = self.__edit_path(path)
        self.data = pd.DataFrame(
            columns=['path', 'dir', 'total_inner_dirs', 'total_inner_files', 'inner_types']
        )
        self.i = 0
    
    def __iterate(self, path):
        for dir_path, inner_dir_names, inner_file_names in os.walk(path):
            return dir_path, inner_dir_names, inner_file_names
    
    def __edit_path(self, path):
        return path.replace('\\', '/')
    
    def __count_content(self, dir_path, inner_dir_names, file_names):
        dir_path = self.__edit_path(dir_path)
        dir_name = dir_path.split('/')[-1]
        dir_path = '/'.join(dir_path.split('/')[:-1])
        
        self.data.loc[self.i, 'path'] = dir_path
        self.data.loc[self.i, 'dir'] = dir_name
        self.data.loc[self.i, 'total_inner_dirs'] = len(inner_dir_names)
        self.data.loc[self.i, 'total_inner_files'] = len(file_names)
        self.data.loc[self.i, 'inner_types'] = ', '.join(set([f".{name.split('.')[-1]}" for name in file_names]))
        
        self.i += 1
    
    def  __print_content(self, dir_path, inner_dir_names, inner_file_names):
        dir_path = dir_path.replace('\\', '/')
        
        print(dir_path)
        print('-' * len(dir_path))
        
        print('\n{:25s}{}'.format('Total directories:', len(inner_dir_names)))
        print('{:25s}{}'.format('Total files:', len(inner_file_names)))
        
        if len(inner_dir_names) > 0:
            print('{:25s}{}'.format('Directories:', ' | '.join(inner_dir_names)))
        
        print('\n')
    
    def get_dir_content(self):
        for dir_path, inner_dir_names, inner_file_names in os.walk(self.path):
            if len(inner_file_names) > 0:
                self.__count_content(*self.__iterate(dir_path)) # __print_content
            else:
                self.__count_content(dir_path, inner_dir_names, inner_file_names) # __print_content
    
    def count_classes(self, inner_dir_name):
        dir_names = self.data[self.data['path'].str.count(inner_dir_name) > 0]['dir'].unique()
        total = 0

        for name in dir_names:
            if self.path.split('/')[-1] not in name:
                total += 1
        
        print('Total unique classes:', total)