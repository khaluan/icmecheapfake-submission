import os
import json

import sys
sys.path.append('../Config')
from Config.config import *
import pandas as pd

def read_data(task_name):
    if task_name == 'task1':
        return pd.read_json(os.path.join(ANNOTATION_DIR, 'test_data.json'), lines=True)
        # with open(, 'r') as file:
        #     content = file.readlines()
        # content = list(map(json.loads, content))
        # return content
    elif task_name == 'task2':
        return pd.read_json(os.path.join(ANNOTATION_DIR, 'task_2.json'))
