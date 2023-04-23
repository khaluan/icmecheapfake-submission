RAW_DIR = '/root/Methods/icmecheapfake-submission/Output/Context_plain_task2'
ANNOTATION_DIR = '/root/Methods/icmecheapfake-submission/Dataset'
OUTPUT_DIR = '/root/Methods/icmecheapfake-submission/Output/Context_EL_task2'
from .config import *
from os.path import join, isfile
from os import listdir
import re
from itertools import groupby

import json
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def get_image_id(filename: str) -> int:
    return int(re.search('([0-9]+)_[0-9]+\.txt', filename).group(1))

def get_image_id_from_local_path(filename: str) -> int:
    return int(re.search('([0-9]+)\.', filename).group(1))


def main(task_name):
    context_files = [join(RAW_DIR[task_name], file) for file in listdir(RAW_DIR[task_name]) if isfile(join(RAW_DIR[task_name], file))]
    context_files = sorted(context_files, key=get_image_id)

    context_files_grouped_by_image_id = {key: list(val) for key, val in groupby(context_files, key = get_image_id)}
    # print(len(context_files_grouped_by_image_id))
    # print(context_files_grouped_by_image_id[1])

    test_data = pd.read_json(join(ANNOTATION_DIR, INPUT_FILE[task_name]))
    filenames = test_data['img_local_path'].to_list()
    # print(filenames)
    filenames = [get_image_id_from_local_path(name) for name in filenames]
    # print(filenames)
    from coref import resolve_references
    from entity_linking import entity_linking

    def parse_entity(row, col_name):
        return entity_linking(row[col_name])

    def union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    test_data['caption_entities'] = test_data.progress_apply(lambda x: parse_entity(x, 'caption'), axis=1)

    # print(test_data.head(5))

    def intersect(list1, list2):
        list1 = [u[1] if u[1] else u[0] for u in list1]
        list2 = [u[1] if u[1] else u[0] for u in list2]
        # print(list1, list2)
        return len(set(list1).intersection(set(list2))) > 0

    for i in tqdm(range(len(filenames))):
        if context_files_grouped_by_image_id.get(filenames[i], None):
            # print(context_files_grouped_by_image_id[i])
            for name in context_files_grouped_by_image_id.get(filenames[i], []):
                # print(f'Processing {name}')
                with open(name, 'r', encoding='utf8') as file:
                    content = json.load(file)
            
                context = content['context']
                if context == '':
                    continue

                # context_resolved = resolve_references(context)
                context_resolved = context

                new_context = []
                for sentence in context_resolved.split('\n'):
                    if intersect(entity_linking(sentence + "."), test_data.iloc[i].caption_entities):
                        new_context.append(sentence)

                content['context'] = '\n'.join(new_context)
                # Change output dir
                new_filename = name.replace(RAW_DIR[task_name], OUTPUT_DIR[task_name])
                with open(new_filename, 'w+', encoding='utf8') as file:
                    json.dump(content, file)