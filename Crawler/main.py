import sys
sys.path.append('../Config')

import logging
from unittest import result
from Config.config import *
from .util import read_data, request
from tqdm import tqdm
from .rev_search import reverse_image_search_try_catch
from .preprocess import preprocess_post
import json
from .crawler import crawl_context, crawl_context_raw, parse_content, require_Javascript
import time
from os.path import exists, join, isfile 
from os import listdir, remove, stat
import re

main_log = logging.getLogger('Main log')
FORMAT = logging.Formatter('%(levelname)s - %(message)s - Post %(i)s - Context %(j)s - %(post_url)s')
LOG_FILE_HANDLER = logging.FileHandler('log/main.log', mode = 'w+')
LOG_FILE_HANDLER.setFormatter(FORMAT)
main_log.addHandler(LOG_FILE_HANDLER)

parse_log = logging.getLogger('Parse log')
FORMAT = logging.Formatter('%(levelname)s - %(message)s - Post %(i)s - Context %(j)s - %(post_url)s')
LOG_FILE_HANDLER = logging.FileHandler('log/parse.log', mode = 'w+')
LOG_FILE_HANDLER.setFormatter(FORMAT)
parse_log.addHandler(LOG_FILE_HANDLER)

def get_image(filename):
    return int(re.search('/.*([0-9]+)', filename).group(1))


def crawler_main(task_name):
    data = read_data(task_name)
    images_id = [get_image(name['img_local_path']) for name in data]
    # print(images_id)
    
    with open(f'./Output/url_{task_name}.txt', 'w+', encoding='utf8') as file:
        for datapoint in tqdm(data):
            try:
                result = ''
                result = reverse_image_search_try_catch(datapoint['img_local_path'])
                result = preprocess_post(result, remove_irrelevant=True)
            except Exception as e:
                # print(e)
                pass
            file.write(str(result))
            file.write('\n')
        


    # Stage 2:         
    with open(f'./Output/url_{task_name}.txt', 'r', encoding='utf8') as input_file:
        content = input_file.readlines()
    # print(len(content))

    # with open('../log/reverse_full.log', 'r', encoding='utf8') as log_file:
    #     log = log_file.readlines()
    # log = list(map(lambda x: x.split(' - ')[-1], log))

    for i in tqdm(range(0, len(data))):

        row = eval(content[i])
        
        for j, post in enumerate(row):
            # start_time = time.time()
            try:
                context_req, context_sel = '', ''
                if post['lang'] == 'en':
                    context_req, context_sel = crawl_context_raw(post)
                with open(f'Output/Context_raw_{task_name}/{images_id[i]}_{j}.txt', 'w+', encoding='utf8') as output_file:
                    output_file.write(post['post_url'] + '\n')
                    output_file.write((context_req))
                # with open(f'../Output/Context_raw/{i}_{j}_sel.txt', 'w+', encoding='utf8') as output_file:
                #     output_file.write((context_sel))

            except Exception as e:
                main_log.error(f'Error {e}', extra = {**post, **({'i':i, 'j':j})})
                main_log.error('Crawl failed', extra = {**post, **({'i':i, 'j':j})})
            # end_time = time.time()
            # print(f'Elapsed time: {end_time - start_time}')
        # break

    # Stage 3
    for i in tqdm(range(len(content))):
        row = eval(content[i])
        
        
        for j, post in enumerate(row):
            # start_time = time.time()
            try:
                if exists(join(RAW_DIR[task_name], f'{images_id[i]}_{j}.txt')):
                    with open(join(RAW_DIR[task_name], f'{images_id[i]}_{j}.txt'), 'r', encoding='utf8') as file:
                        url = file.readline()
                        response = file.read()
                    context = parse_content(response, post)
                    context['url'] = url
                if require_Javascript(context):
                    continue
                with open(f"{CONTEXT_DIR[task_name]}/{images_id[i]}_{j}.txt", 'w+', encoding='utf8') as file:
                    json.dump(context, file)

            except Exception as e:
                parse_log.error(f'Error {e}', extra = {**post, **({'i':i, 'j':j})})
                parse_log.error('Crawl failed', extra = {**post, **({'i':i, 'j':j})})
            
    # Stage 4: remove empty context
    filenames = [join(CONTEXT_DIR[task_name], f) for f in listdir(CONTEXT_DIR[task_name]) if isfile(join(CONTEXT_DIR[task_name], f))]   
    for filename in filenames:
        try:
            if stat(filename).st_size == 0:
                remove(filename)
            with open(filename, 'r', encoding='utf8') as file:
                content = eval(file.read())
                if content['heading'] == '' and content['context'] == '':
                    remove(filename)
        except Exception as e:
            # print(e)
            pass
            # print(filename)
