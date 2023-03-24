import torch
from .QA import answer
import sys
sys.path.append('../Config')
from Config.config import *
from .utils import read_data

import json
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

from os.path import join, isfile
from os import listdir
from itertools import groupby
import re
import numpy as np
import random
from sklearn.metrics import *


def main(task_name):
    # Use Q&A model to extract answer
    def get_image_id(filename: str) -> int:
        return int(re.search('([0-9]+)_[0-9]+\.txt', filename).group(1))

    context_files = [join(CONTEXT_REFINED_DIR[task_name], file) for file in listdir(CONTEXT_REFINED_DIR[task_name]) if isfile(join(CONTEXT_REFINED_DIR[task_name], file))]
    context_files = sorted(context_files, key=get_image_id)

    context_files_grouped_by_image_id = {key: list(val) for key, val in groupby(context_files, key = get_image_id)}

    # test_data = pd.read_json(join(ANNOTATION_DIR, 'test_data.json'), lines=True)
    test_data = read_data(task_name)

    test_data['context_filename'] = pd.DataFrame([context_files_grouped_by_image_id]).T
    test_data['context_filename'] = [ [] if x is np.NaN else x for x in test_data['context_filename'] ]

    def process(row):
        context_filename = row.context_filename
        answers = []
        try:
            for filename in context_filename:
                with open(filename, 'r', encoding='utf8') as ifile:
                    context = ifile.read()
                if context == '':
                    continue
                context = ' '.join(context.split('\n'))
                answers.append(answer(context, row.caption1, row.caption2))
        except Exception as e:
            print(e)
            print(filename)
        return answers

    test_data['answers'] = test_data.progress_apply(process, axis=1)
    print(test_data.head(3))

    # Use SBERT-WK to assess the answers
    from .sbert import sbert

    def sim(row):
        return [sbert(sentences) for sentences in row.answers if len(sentences) == 2 ]

    test_data['sim'] = test_data.progress_apply(sim, axis=1)

    def infer(row, thresh):
        pred = [i < thresh for i in row.sim]
        return pred

    def get_mode(row):
        li = row.pred
        if li == []:
            return True
        # li = infer(li)
        val, count = np.unique(li, return_counts=True)

        if len(count) == 2 and count[0] == count[1]:
            return False
        else:
            return val[np.argmax(count)]

    prec, recall, acc = [], [], []



    test_data['pred'] = test_data.progress_apply(lambda row: infer(row, thresh), axis=1)
    test_data['context_pred'] = test_data.progress_apply(get_mode, axis=1)

    pred = list(test_data['context_pred'])

    ground_truth = list(test_data['context_label'])
    for i, x in enumerate(pred):
        if x is None:
            pred[i] = test_data.iloc[i].bert_base_score < 0.5

    test_data['context_pred'] = pred
    test_data.to_csv(f'./df_answer_{task_name}.csv')
            