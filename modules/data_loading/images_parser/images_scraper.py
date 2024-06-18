'''
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import hashlib
import urllib.request
import io
import os
import time
from pathlib import Path
from tqdm.notebook import tqdm
from PIL import Image


class ImagesScraper():
    
    def __init__(self, url: str, limit: int):
        driver = webdriver.Edge(executable_path=Path('modules/images_parser/msedgedriver.exe'))
        driver.get(url)
        self.driver = driver
        self.limit = limit


    def _scroll_down(self, steps: int) -> None:
        for _ in range(steps):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
    

    def _get_elements(self, query: str, single: bool = False) -> list:
        if single:
            return self.driver.find_element(By.CSS_SELECTOR, query)
        else:
            return self.driver.find_elements(By.CSS_SELECTOR, query)


    def _click_on_element(self, element: webdriver.remote.webelement.WebElement) -> None:
        try:
            self.driver.execute_script("arguments[0].click();", element)
        except Exception:
            #print("Error within a click on an element")
            self.error_clicks += 1


    def _download_image(self, folder_path: str, url: str, idx: int) -> None:
        if os.path.isdir(folder_path) is False:
            os.mkdir(folder_path)
        
        try:
            image_path = os.path.join(folder_path, f'{idx}.jpg')
            urllib.request.urlretrieve(url, image_path)
            self.total += 1

        except Exception as e:
            #print(f"Error within saving of {url} - {e}")
            self.error_downloads += 1
            

    def parse_images(self, image_info: set, query: str, folder_path: str) -> None:
        if len(os.listdir(folder_path)) > 0:
            end = sorted([
                int(name.split('.')[0]) 
                for name in os.listdir(folder_path)
            ])[-1]
        else:
            end = 0

        for i, image in enumerate(tqdm(image_info, desc='Saving')):
            self._download_image(folder_path, image, end + i)
        
        print(f"\nTotal error clicks: {self.error_clicks}")
        print(f"Total error downloads: {self.error_downloads}")
        print(f"Total images downloaded: {self.total}")
        print(f"Saved to {folder_path}")
    

    def _rescrape_images(self):
        self.error_clicks = 0
        self.error_downloads = 0
        self.total = 0