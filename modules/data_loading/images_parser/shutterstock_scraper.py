'''
'''

from tqdm.notebook import tqdm
from .images_scraper import ImagesScraper


class ShutterstockImagesScraper(ImagesScraper):
    
    def __init__(self, url: str = "https://www.shutterstock.com/", limit: int = 100):
        super().__init__(url, limit)


    def _build_query(self, query: str, page: int) -> str:
        return f"https://www.shutterstock.com/search/{query}?page={page}"
    

    def _get_info(self, query: str, pages: int, images_class: str) -> list:
        image_urls = set()

        for page in tqdm(range(1, pages + 1), desc='Clicking'):
            try:
                self.driver.get(self._build_query(query, page))
            except:
                continue
            
            self._scroll_down(5)
                
            links = self._get_elements(f'img.{images_class}')
            #print(f"Total links: {len(links)}")

            for link in links[0:self.limit]:
                image_urls.add(link.get_attribute('src'))

        return image_urls
            

    def scrape_images(self, query: str, pages: int, folder_path: str) -> None:
        super()._rescrape_images()

        image_info = self._get_info(query, pages, images_class='mui-1l7n00y-thumbnail')
        self.parse_images(image_info, query, folder_path)