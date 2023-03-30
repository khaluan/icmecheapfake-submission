import os, io
from config import *
import logging
from secret import *

reverse_log = logging.getLogger('Reverse log')
FORMAT = logging.Formatter('%(levelname)s - %(message)s - %(img_path)s')
LOG_FILE_HANDLER = logging.FileHandler('../log/reverse.log', mode = 'w+')
LOG_FILE_HANDLER.setFormatter(FORMAT)
reverse_log.addHandler(LOG_FILE_HANDLER)

from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './cred.json'
client = vision.ImageAnnotatorClient()

def reverse_image_search_try_catch(img_path):
    with io.open(os.path.join(IMAGE_DIR, img_path), 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content = content)
    response = client.web_detection(image = image)
    annotations = response.web_detection
    result = []
    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            result.append({
                'post_url': page.url,
                'title': page.page_title
            })
    return result

# Example
# img_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e0/180802_%EB%B6%80%EC%82%B0%EB%B0%94%EB%8B%A4%EC%B6%95%EC%A0%9C_%ED%95%98%ED%95%98_1.jpg'
# print(reverse_image_search_try_catch('1.jpg'))
