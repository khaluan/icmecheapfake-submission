import requests
import certifi
import sys
sys.path.append('../Config')

from Config.config import *

import os
import json
import logging
import pandas as pd

request_log = logging.getLogger('Request log')
FORMAT = logging.Formatter('%(levelname)s - %(message)s - %(url)s')
LOG_FILE_HANDLER = logging.FileHandler('log/request.log', mode = 'w+')
LOG_FILE_HANDLER.setFormatter(FORMAT)
request_log.addHandler(LOG_FILE_HANDLER)

def read_data(task_name):
    if task_name == 'task1':
        with open(os.path.join(ANNOTATION_DIR, 'test_data.json'), 'r') as file:
            content = file.readlines()
        content = list(map(json.loads, content))
        return content
    elif task_name == 'task2':
        df = pd.read_json(os.path.join(ANNOTATION_DIR, 'task_2.json'))
        content = df.to_dict('records')
        return content

def request(url):
    try:
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0'}
        response = requests.get(url, headers=header, verify=certifi.where(), allow_redirects=True, timeout=5)

        if response.status_code != 200:
            request_log.error('URL not found', extra={'url': url})
        
        # TODO: Catch capcha case

        return response.text
    except requests.exceptions.ReadTimeout:
        request_log.error('Request timeout', extra={'url':url})
        raise Exception
    except:
        request_log.error('Cannot request to ',extra={'url':url})
        raise Exception