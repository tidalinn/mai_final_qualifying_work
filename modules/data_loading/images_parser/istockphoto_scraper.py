'''
'''

from tqdm.notebook import tqdm
from .shutterstock_scraper import ShutterstockImagesScraper


class IstockphotoImagesScraper(ShutterstockImagesScraper):
    
    def __init__(self, url: str = "https://www.istockphoto.com/ru/", limit: int = 100):
        super().__init__(url, limit)


    def _build_query(self, query: str, page: int) -> str:
        return f"https://www.istockphoto.com/ru/search/2/image-film?phrase={query}&page={page}"
            

    def scrape_images(self, query: str, pages: int, folder_path: str) -> None:
        super()._rescrape_images()

        image_info = self._get_info(query, pages, images_class='yGh0CfFS4AMLWjEE9W7v')
        self.parse_images(image_info, query, folder_path)