from tqdm import tqdm
import pandas as pd
import numpy as np
tqdm.pandas()
from sklearn.metrics import *

def evaluate(df, func):
    df['result'] =  df.progress_apply(lambda x:func(x), axis=1)
    df['predict'] =  df['result'].apply(lambda x:x[0])
    df['method'] =  df['result'].apply(lambda x:x[1])
    confusion_matrix = pd.crosstab(df['predict'], df['context_label'], rownames=['Predicted'], colnames=['Actual'])
    print(np.unique(df['predict']))
    result = (confusion_matrix[0][0]+confusion_matrix[1][1])/len(df)
    print('Accuracy:', result)
    print('Recall:', recall_score( df['context_label'], df['predict']))
    print('Precision:', precision_score( df['context_label'], df['predict']))

    method_acc = df.groupby('method').apply(lambda g: \
        ((g['context_label']==g['predict']).sum() / len(g),len(g) ))
    print(method_acc)
    return confusion_matrix, result, method_acc

def predict_baseline(row):
    if row['iou']>0.5:
        return (row['bert_base_score']<0.5, 'BERT')
    return [False,'COSMOS']

def predict_final(row):

    if row['iou']>0.5:
        return (row['bert_base_score']<0.5, 'COSMOS')
    else:
        if row['nli'] and row['bert_base_score']<0.5:
            return (True, 'DOCNLI')
        return (False, 'LAST HIT')

def predict_combine_bert(row):
    if (row['context_pred']) ^ (row['bert_base_score'] < 0.5):
        return (False, "CONTEXT")
    return (True, "CONTEXT")


def predict_context(row):
    return (row['context_pred'], 'CONTEXT')