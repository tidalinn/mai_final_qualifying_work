'''
'''

from selenium.webdriver.common.by import By
from tqdm.notebook import tqdm
from .images_scraper import ImagesScraper


class YandexImagesScraper(ImagesScraper):
    
    def __init__(self, url: str = "https://yandex.ru/images", limit: int = 100):
        super().__init__(url, limit)


    def _build_query(self, query: str) -> str:
        #time.sleep(15) # to manually resolve the captcha
        return f"https://yandex.ru/images/search?from=tabbar&text={query}"
    

    def _get_image(self, button_image_class: str, urls: list) -> None:
        url = self.driver.find_element(By.CSS_SELECTOR, f'div.{button_image_class} a')
        urls.add(url.get_attribute('href'))
    

    def _get_info(self, query: str, steps: int, thumbs_class: str, button_image_class: str, button_next_class: str, button_prev_class: str) -> list:
        image_urls = set()

        self.driver.get(self._build_query(query))
        self._scroll_down(steps)
        
        thumbs = self._get_elements(f'img.{thumbs_class}')
        #print(f"Total thumbs: {len(thumbs)}")

        for thumb in tqdm(thumbs[0:self.limit], desc='Clicking'):
            self._click_on_element(thumb)            
            self._get_image(button_image_class, image_urls)

            relatives = self._get_elements(f'span img.{thumbs_class}')

            for relative in relatives:
                self._click_on_element(relative)
                self._get_image(button_image_class, image_urls)

                self._click_on_element(self.driver.find_element(By.CSS_SELECTOR, f'button.{button_next_class}'))
                self._click_on_element(self.driver.find_element(By.CSS_SELECTOR, f'button.{button_prev_class}'))

        return image_urls
            

    def scrape_images(self, query: str, steps: int, folder_path: str) -> None:
        super()._rescrape_images()

        image_info = self._get_info(
            query, 
            steps, 
            thumbs_class='ContentImage-Image.ContentImage-Image_clickable', 
            button_image_class='OpenImageButton',
            button_next_class='ImagesViewer-ButtonNext',
            button_prev_class='ImagesViewer-ButtonPrev'
        )

        self.parse_images(image_info, query, folder_path)