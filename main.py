file = open("Output.txt", 'w+')
task_name = 'task1'
print("Task 1", file=file)
print("Step 1: Started to crawl context", file=file)
from Crawler import main
main.crawler_main(task_name)
print("Crawling finished", file=file)

print("Step 2: Use text summary for text crawling", file=file)
from Summarization import main
main.summary(task_name)
print("Summary finished", file=file)


print("Step 3: QA", file=file)
from QA.main import main as QA_main
QA_main(task_name)
print("Step 3 finished", file=file)

print("Step 4: Use DocNLI model on the collected context", file=file)
import DocNLI
DocNLI.main(task_name)
print("Step 4 completed", file=file)

print("Step 5: COSMOS baseline", file=file) 
import sys 
sys.path.append('./COSMOS')
from COSMOS import evaluate_ooc

print('Running COSMOS', file=file)
evaluate_ooc.main(None)

print("Step 6: Boosting and evaluation", file=file)
import pandas as pd
from evaluate_utils import *
df = pd.read_csv('df_answer_task1.csv')
cosmos_iou = pd.read_csv('pred_contexts.txt', header=None)
cosmos_iou.columns = ['iou']
df = pd.concat([df, cosmos_iou['iou']], axis=1)

docnli = eval(open('docnli_task1.txt', 'r').read())
df['nli'] = docnli

print('Evaluating task 1', file=file)
confusion_matrix, result, method_acc, pred = evaluate(df, predict_final)
# print("Confusion matrix")
#print("Acc", result)
print("Accuracy task 1:", result)
pred = [int(x) for x in pred.to_list()]
print(pred)


# ######################## Task 2 ############################
task_name = 'task2'

print("Task 2", file=file)
print("Step 1: Started to crawl context", file=file)
from Crawler import main
main.crawler_main(task_name)
print("Crawling finished", file=file)

print("Step 2: Use text summary for text crawling", file=file)
from Summarization import main
main.summary(task_name)
print("Summary finished", file=file)

print("Step 3: Use DocNLI model on the collected context", file=file)
import DocNLI
DocNLI.main(task_name)
print("Step 3 completed", file=file)

print("Step 4: Evaluate", file = file)
from util import read_data
import json
from sklearn.metrics import *
print("Evaluating task 2")
pred = json.load(open('docnli_task2.txt'))
df = read_data(task_name)
ground_truth = df['genuine'].to_list()
print("Accuracy task 2:", accuracy_score(ground_truth, pred))
print(pred)
