# Use Bert model to do NLI task
# UAE: Universal Adversarial Examples for Textual Entailment
import os,sys
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForTokenClassification, AdamW, BertConfig
from transformers import BertForSequenceClassification

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset

from time import sleep
from tqdm import tqdm
import numpy as np
import random

# load SNLI dataset
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

import argparse
import json
import nltk

def loadDataset(dataset):
    premise = [data[0] for data in dataset]
    hypothesis = [data[1] for data in dataset]
    labels = [data[2] for data in dataset]
    return premise,hypothesis,labels
# POS of setence using NLTK
def get_POS(sentence):
    pos = nltk.pos_tag(nltk.word_tokenize(sentence))
    pos = [p[1] for p in pos]
    return pos
# mask tokens in sentence from 3 words in head
def preprocessing_Mask(hypothesis,transition,mask_num):
    # random select 1 words in transition_list
    new_hypothesis = []
    for sentence in hypothesis:
        # substitute = random.sample(transition_list,1)
        substitute = [transition]
        # replace 1 words in head with [MASK]
        substitute = substitute[0].split(" ")
        sentence = sentence.split()
        sentence[:mask_num] = ["[MASK]"]*mask_num
        sentence = substitute + sentence
        sentence = " ".join(sentence)
        new_hypothesis.append(sentence)
    return new_hypothesis

def tokenize(premise,hypothesis):
    input_ids = []
    attention_masks = []
    for s1,s2 in zip(premise,hypothesis):
        encoded_pair = tokenizer.encode_plus(
                s1,                       # 句子1
                s2,                    # 句子2
                max_length=512,            # 设置最大长度
                truncation=True,               # 超过最大长度则截断
                padding='max_length',          # 不足最大长度则填充
                add_special_tokens=True,       # 添加特殊的token (CLS, SEP)
                return_attention_mask=True,    # 返回attention mask
                return_tensors='pt',           # 返回PyTorch的tensor
            )
        input_ids.append(encoded_pair['input_ids'])
        attention_masks.append(encoded_pair['attention_mask'])
    return input_ids,attention_masks

# load SNLI dataset from data folder
# there are snli_1.0_dev.jsonl, snli_1.0_test.jsonl, snli_1.0_train.jsonl in data folder
def load_Jsondata(path):
    #load jsonl file
    dev_path = os.path.join(path,'snli_1.0_dev.jsonl')
    test_path = os.path.join(path,'snli_1.0_test.jsonl')
    train_path = os.path.join(path,'snli_1.0_train.jsonl')
    # load data
    dev_data = []
    with open(dev_path,'r') as f:
        for line in f:
            dev_data.append(json.loads(line))
        #load fields in snli1.0 jsonl
    dev_data = [data for data in dev_data if data['gold_label'] != '-']
    dev_data = [(data['sentence1'],data['sentence2'],data['gold_label']) for data in dev_data]

    test_data = []
    with open(test_path,'r') as f:
        for line in f:
            test_data.append(json.loads(line))
        #load fields in snli1.0 jsonl
    test_data = [data for data in test_data if data['gold_label'] != '-']
    test_data = [(data['sentence1'],data['sentence2'],data['gold_label']) for data in test_data]

    train_data = []
    with open(train_path,'r') as f:
        for line in f:
            train_data.append(json.loads(line))
        #load fields in snli1.0 jsonl
    train_data = [data for data in train_data if data['gold_label'] != '-']
    train_data = [(data['sentence1'],data['sentence2'],data['gold_label']) for data in train_data]
    
    return dev_data,test_data,train_data

if __name__ == "__main__":
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')

    # tokenizer initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased',num_labels=3)
    dev_data,test_data,train_data = load_Jsondata("source/data/SNLI/snli_1.0")

    # load test data
    premise,hypothesis,labels = loadDataset(test_data)
    # map labels to 0,1,2
    labels = [0 if label == 'entailment' else 1 if label == 'neutral' else 2 for label in labels]

    with open('source/data/SNLI/transition_list.txt','r') as f:
        transition_list = f.readlines()
        transition_list = [transition.strip() for transition in transition_list]
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load('SNLI_BERT.pth'))
    model = model.to(device)
    model.eval()
    torch.no_grad()
    count = 0
    for transition in transition_list:
        hypothesis = preprocessing_Mask(hypothesis,transition,0)
        print(transition)
        get_POS(hypothesis[0])
        input_ids,attention_masks = tokenize(premise,hypothesis)
        input_ids = torch.cat(input_ids,dim=0)
        attention_masks = torch.cat(attention_masks,dim=0)
        labels = torch.tensor(labels)

        test_dataset = TensorDataset(
            input_ids,
            attention_masks,
            labels
        )

        test_dataloader = DataLoader(
            test_dataset,
            sampler = RandomSampler(test_dataset),
            batch_size = 32
        )

        total_accuracy = 0
        for step,batch in enumerate(test_dataloader):
            batch = [r.to(device) for r in batch]
            outputs = model(
                batch[0],
                attention_mask=batch[1]
            )
            logits = outputs.logits
            count += 32
            print(count)

            #compare prediction and labels to get accuracy with batch
            prediction = torch.argmax(logits,dim=1)
            accuracy = torch.sum(prediction == batch[2]).item() / len(prediction)

            # release outputs on GPU
            del outputs
            del logits
            del batch
            total_accuracy += accuracy
        print(total_accuracy/len(test_dataloader))

# 5. metrics for evaluation

