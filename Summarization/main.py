from .summary import summary_text
import sys
sys.path.append('../Config')
from Config.config import *
from os.path import exists, join, isfile 
from os import listdir, remove, stat
from tqdm import tqdm

def summary(task_name):
    filenames = [f for f in listdir(CONTEXT_DIR[task_name]) if isfile(join(CONTEXT_DIR[task_name], f))]   
    for filename in tqdm(filenames):
        try:
            with open(join(CONTEXT_DIR[task_name], filename), 'r', encoding='utf8') as file:
                content = eval(file.read())
            summary = summary_text(content['heading'] + content['context'])
            with open(join(CONTEXT_REFINED_DIR[task_name], filename), 'w+', encoding='utf8') as file:
                file.write(summary)
            
        except Exception as e:
            # print(e)
            # print(filename)
            pass