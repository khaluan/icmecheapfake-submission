from typing import Dict, List
from bs4 import BeautifulSoup
from .util import request
import requests
import os
import sys
sys.path.append('../Config')

from Config.config import *
import logging
import re
from .preprocess import preprocess_post

reverse_log = logging.getLogger('Reverse log')
FORMAT = logging.Formatter('%(levelname)s - %(message)s - %(img_path)s')
LOG_FILE_HANDLER = logging.FileHandler('log/reverse.log', mode = 'w+')
LOG_FILE_HANDLER.setFormatter(FORMAT)
reverse_log.addHandler(LOG_FILE_HANDLER)

# SEARCH_URL = 'https://www.google.com/searchbyimage?hl=en-US&image_url='
SEARCH_BY_IMG_URL = 'http://www.google.hr/searchbyimage/upload'
BASE_URL = 'https://www.google.com'

def get_all_post_in_current_page(page_html: str, first_page=True) -> List[Dict]:
    """
        Returns: a Lisit of relevant posts url and their title
            Each object in the list consists of 2 properties
                post_url: str, url of the post
                title   : str, name of the post title (to identify the language used in the post)
        
        Parameter page_html: The html response of a page for crawling
    """
    soup = BeautifulSoup(page_html, 'html.parser')

    sibling = None
    if first_page:
        HEADER_CLASS = 'normal-header'
        header = soup.find('div', HEADER_CLASS)
        sibling = header.next_sibling
    else:
        POST_CLASS = 'SC7lYd'
        sibling = soup.find('div', {'jscontroller' : POST_CLASS})

    result = []    
    while True:
        try:
            if first_page:
                result.append(
                    {
                    'post_url': sibling.div.div.a.get('href'),
                    'title'   : sibling.div.div.a.h3.text
                    }
                    )
            else:
                result.append(
                    {
                    'post_url': sibling.div.div.div.a.get('href'),
                    'title'   : sibling.div.div.div.a.h3.text
                    }
                    )
            sibling = sibling.next_sibling

        except Exception as e:
            # print(e)
            break
    return result


def search_image(search_url: str, img_path: str) -> List[Dict]:
    """
        Returns: a Lisit of relevant posts url and their title
            Each object in the list consists of 2 properties
                post_url: str, url of the post
                title   : str, name of the post title (to identify the language used in the post)
        
        Parameter search_url: The search url retrieved via get_search_url function
                    img_path: Path to the lookup image
    """
    response = request(search_url)
    soup = BeautifulSoup(response, 'html.parser')

    result = []
    
    ID_NAME = 'result-stats'
    search_stat = soup.find('div', {"id": ID_NAME}).text

    NUM_REGEX = '[0-9.]+'
    result_cnt = int(re.findall(NUM_REGEX, search_stat)[0].replace('.', ''))
    
   
    result_cnt = min(result_cnt, MAX_POST_PER_SAMPLE)
    reverse_log.warning(f'Found {result_cnt} result', extra={'img_path': img_path})

    pages_url = []
    PAGE_JSNAME = 'TeSSVd'
    pagination_nav = list(soup.find('tr', {"jsname": PAGE_JSNAME}).children)
    for child in pagination_nav:
        try:
            pages_url.append(child.a.get('href'))
        except:
            pass
    if len(pages_url) > 0:
        pages_url.pop() # Remove the url for next button
    reverse_log.warning(f'Found {len(pages_url)} pages', extra={'img_path': img_path})

    result = get_all_post_in_current_page(response)
    while True:
        if len(result) > result_cnt:
            reverse_log.warning(f'Actual found {len(result)} result', extra={'img_path': img_path})
            return result[:result_cnt]
        elif len(pages_url) == 0:
            break
        else:
            result += get_all_post_in_current_page(request(BASE_URL + pages_url[0]), first_page=False) # Remember to set first_page = False
            pages_url.pop(0) # remove the first element in the queue

    # CLASS_NAME = 'isv-r PNCib MSM1fd BUooTd'
    # elements = soup.find_all('div', CLASS_NAME)
    
    # result = []
    # for div in elements:
    #     element = div.a.next_sibling
    #     img = div.a.img
    #     result.append(
    #         {
    #             'post_url' : element.get('href'),
    #             'title'    : element.get('title'),
    #             'img_url'  : img.get('src')
    #         }
    #     )
    #     # break
    # return result
    reverse_log.warning(f'Actual found {len(result)} result', extra={'img_path': img_path})
    return result

def get_search_url(img_path: str) -> str:
    '''
        Returns: similar image search url, used for search function

        Parameter image_url: the url of the image
    '''
    # Copied from https://stackoverflow.com/questions/23270175/google-reverse-image-search-using-post-request
    searchUrl = 'http://www.google.com/searchbyimage/upload'
    multipart = {'encoded_image': (img_path, open(os.path.join(IMAGE_DIR, img_path), 'rb')), 'image_content': ''}
    response = requests.post(searchUrl, files=multipart, allow_redirects=False)
    fetchUrl = response.headers['Location']
    return fetchUrl

def reverse_image_search(img_path):
    try:
        search_url = get_search_url(img_path) 
        # print(search_url)   
        return search_image(search_url, img_path)
    except Exception as e:
        reverse_log.error('Search failed', extra={'img_path': img_path})
        reverse_log.error(e, extra={'img_path': img_path})
        raise Exception

def reverse_image_search_try_catch(img_path):
    for _ in range(3):
        try:
            return reverse_image_search(img_path)
        except Exception as e:
            continue
    return []

# Example
# img_url = 'https://upload.wikimedia.org/wikipedia/commons/e/e0/180802_%EB%B6%80%EC%82%B0%EB%B0%94%EB%8B%A4%EC%B6%95%EC%A0%9C_%ED%95%98%ED%95%98_1.jpg'
# (reverse_image_search_try_catch('22.jpg'))
