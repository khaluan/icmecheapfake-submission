import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaModel
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from scipy.special import softmax

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
from Config.config import *
from sklearn.metrics import *
from QA.utils import read_data


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, image_id = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.image_id = image_id
    
    def __str__(self):
        return f'Text_a: {self.text_a}, text_b: {self.text_b}, image_id: {self.image_id}'

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_MNLI_train_and_dev(self, train_filename, dev_filename_list):
        '''
        classes: ["entailment", "neutral", "contradiction"]
        '''
        examples_per_file = []
        for filename in [train_filename]+dev_filename_list:
            examples=[]
            readfile = codecs.open(filename, 'r', 'utf-8')
            line_co=0
            for row in readfile:
                if line_co>0:
                    line=row.strip().split('\t')
                    guid = "train-"+str(line_co-1)
                    # text_a = 'MNLI. '+line[8].strip()
                    text_a = line[8].strip()
                    text_b = line[9].strip()
                    label = line[-1].strip() #["entailment", "neutral", "contradiction"]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
            readfile.close()
            print('loaded  MNLI size:', len(examples))
            examples_per_file.append(examples)
        dev_examples = []
        for listt in examples_per_file[1:]:
            dev_examples+=listt
        return examples_per_file[0], dev_examples #train, dev

    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def get_image_id(filename: str) -> int:
    return int(re.search('([0-9]+)_[0-9]+\.txt', filename).group(1))

def get_image(filename):
    return int(re.search('/([0-9]+)', filename).group(1))

def load_DocNLI(task_name, context_files_grouped_by_image_id, hypo_only=False):
    data = read_data(task_name)
    examples = []
    if task_name == 'task1':
        for _, dic in data.iterrows():
            # print(dic.img_local_path)
            i = get_image(dic.img_local_path)
            for filename in context_files_grouped_by_image_id.get(i, []):
                with open(filename, 'r', encoding='utf8') as file:
                    # content = json.load(file)
                    # context = content['context']
                    context = file.read()
                    hypothesis1 = data.iloc[i].caption1
                    hypothesis2 = data.iloc[i].caption2
                    label = 0
                    examples.append(InputExample(guid='ex', text_a=context, text_b=hypothesis1, label='entailment', image_id = i))
                    examples.append(InputExample(guid='ex', text_a=context, text_b=hypothesis2, label='not_entailment', image_id = i))
        return examples
    elif task_name == 'task2':
        for id, dic in data.iterrows():
            # print(dic.img_local_path)
            i = get_image(dic.img_local_path)
            for filename in context_files_grouped_by_image_id.get(i, []):
                with open(filename, 'r', encoding='utf8') as file:
                    context = file.read()
                    hypothesis1 = data.iloc[id].caption
                    hypothesis2 = ''
                    label = 0
                    examples.append(InputExample(guid='ex', text_a=context, text_b=hypothesis1, label='entailment', image_id = i))
                    examples.append(InputExample(guid='ex', text_a=context, text_b=hypothesis2, label='not_entailment', image_id = i))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def evaluation(dev_dataloader, device, model):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    gold_label_ids = []
    # print('Evaluating...')
    for input_ids, input_mask, segment_ids, label_ids in dev_dataloader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        gold_label_ids+=list(label_ids.detach().cpu().numpy())

        with torch.no_grad():
            logits = model(input_ids, input_mask)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        nb_eval_steps+=1
        # print('eval_steps:', nb_eval_steps, '/', len(dev_dataloader))

    preds = preds[0]

    pred_probs = softmax(preds,axis=1)
    pred_label_ids = list(np.argmax(pred_probs, axis=1))

    gold_label_ids = gold_label_ids
    assert len(pred_label_ids) == len(gold_label_ids)
    # print('gold_label_ids:', gold_label_ids)
    # print('pred_label_ids:', pred_label_ids)
    # f1 = f1_score(gold_label_ids, pred_label_ids, pos_label= 0, average='binary')
    # return f1
    return pred_label_ids

class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size, pretrain_model_dir, bert_hidden_dim):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single

class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Args:
    def __init__(self, do_lower_case = True, 
                    max_seq_length = 512,
                    seed = 42, 
                    eval_batch_size = 32) -> None: 
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.seed = seed
        self.eval_batch_size = eval_batch_size
        pass

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def unflatten(task_name, list, pattern, images_name):
    images_id = [get_image(name) for name in images_name]    
    # print(images_id)
    # print(pattern)
    cnt = 0
    new_list = []
    for i in images_id:
        l = 2 * len(pattern.get(i, []))
        new_list.append(list[cnt:cnt + l])
        cnt += l
        # print(i, l, cnt)
    # print('cnt', cnt)
    assert cnt == len(list)
    return new_list

def infer(task_name, li):
    pred = []
    if task_name == 'task1':
        for i in range(0, len(li), 2):
            pred.append(abs(li[i] - li[i+1]))
        return pred
    elif task_name == 'task2':
        for i in range(0, len(li), 2):
            pred.append(li[i])
        return pred

def get_mode(task_name, li):
    if li == []:
        return 0
    li = infer(task_name, li)
    val, count = np.unique(li, return_counts=True)

    if len(count) == 2 and count[0] == count[1]:
        return 1
    else:
        return val[np.argmax(count)]


def main(task_name):
    
    context_files = [join(CONTEXT_REFINED_DIR[task_name], file) for file in listdir(CONTEXT_REFINED_DIR[task_name]) if isfile(join(CONTEXT_REFINED_DIR[task_name], file))]
    context_files = sorted(context_files, key=get_image_id)
    context_files_grouped_by_image_id = {key: list(val) for key, val in groupby(context_files, key = get_image_id)}

    args = Args()
    device = torch.device('cuda')
    label_list = ["entailment", "not_entailment"]#, "contradiction"]
    num_labels = len(label_list)

    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_mode = 'classification'
    # print('num_labels:', num_labels,  ' test size:', len(test_examples))

    bert_hidden_dim = 1024
    pretrain_model_dir = 'roberta-large'
    model = RobertaForSequenceClassification(num_labels, pretrain_model_dir, bert_hidden_dim)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.load_state_dict(torch.load('./cache/DocNLI.pretrained.RoBERTA.model.pt', map_location=device))

    model.to(device)

    test_examples = load_DocNLI(task_name, context_files_grouped_by_image_id)
    # print(test_examples[1])
    # print(test_examples[3])
    test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    final_test_performance = evaluation(test_dataloader, device, model)

    df = read_data(task_name)
    ground_truth = df['context_label'].to_list() if task_name == 'task1' else df['genuine']
    # print(len(final_test_performance))
    nli_grouped_by_image_id = (unflatten(task_name, final_test_performance, context_files_grouped_by_image_id, df['img_local_path']))
    # print(len(nli_grouped_by_image_id))
    pred = [get_mode(task_name, li) for li in nli_grouped_by_image_id]
    # print(pred)
    import collections
    freq = collections.Counter(pred)
    # print(freq)

    for i, x in enumerate(pred):
        if x is None:
            pred[i] = 1

    # print(confusion_matrix(ground_truth, pred, labels = [0, 1]))
    # print(f'precision: {precision_score(ground_truth, pred)}')
    # print(f'recall: {recall_score(ground_truth, pred)}')
    # print(f'accuracy: {accuracy_score(ground_truth, pred)}')
    # print(f'f1: {f1_score(ground_truth, pred)}')

        
    with open(f"./docnli_{task_name}.txt", 'w+') as file:
        json.dump(pred, file, cls=NpEncoder)
