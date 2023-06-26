# Use Bert model to do NLI task
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

# tokenizer initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

dev_data,test_data,train_data = load_Jsondata("source/data/SNLI/snli_1.0")

# 1. load SNLI data
# premise = [data[0] for data in train_data]
# hypothesis = [data[1] for data in train_data]
# labels = [data[2] for data in train_data]
# # map labels to 0,1,2
# labels = [0 if label == 'entailment' else 1 if label == 'neutral' else 2 for label in labels]

# input_ids,attention_masks = tokenize(premise,hypothesis)

# # convert to tensor
# input_ids = torch.cat(input_ids,dim=0)
# attention_masks = torch.cat(attention_masks,dim=0)
# labels = torch.tensor(labels)

# dataset = TensorDataset(
#     input_ids,
#     attention_masks,
#     labels
# )
# # 2. load training data
# train_dataloader = DataLoader(
#     dataset,
#     sampler = RandomSampler(dataset),
#     batch_size = 32
# )

# load test data
premise = [data[0] for data in test_data]
hypothesis = [data[1] for data in test_data]
labels = [data[2] for data in test_data]
# map labels to 0,1,2
labels = [0 if label == 'entailment' else 1 if label == 'neutral' else 2 for label in labels]

input_ids,attention_masks = tokenize(premise,hypothesis)
# map every labels to 0,1,2
# convert to tensor
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

# 3. load pretrained BERT model
config = BertConfig.from_pretrained('bert-base-uncased',num_labels=3)
# model = BertForSequenceClassification(config)
# # 4. fine-tuning BERT model by using SNLI data
# model=model.to(device)
# model.train()
# optimizer = AdamW(model.parameters(),lr=5e-5,eps=1e-8)
# epochs = 3
# for epoch in tqdm(range(epochs)):
#     for step,batch in enumerate(train_dataloader):
#         batch = [r.to(device) for r in batch]
#         outputs = model(
#             batch[0],
#             token_type_ids=None,
#             attention_mask=batch[1],
#             labels=batch[2]
#         )
#         loss = outputs.loss
#         logits = outputs.logits
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         #count loss
#         if step % 100 == 0:
#             print(loss.item())

# torch.save(model.state_dict(), 'SNLI_BERT.pth')
# 4.1 load test data
# 4.2 load fine-tuned BERT model
torch.cuda.empty_cache()
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load('SNLI_BERT.pth'))
model = model.to(device)
model.eval()
# classfier = nn.Sequential(
#     nn.Linear(768,256),
#     nn.ReLU(),
#     nn.Linear(256,3)
# )
# classfier = classfier.to(device)
torch.no_grad()
total_accuracy = 0
for step,batch in enumerate(test_dataloader):
    batch = [r.to(device) for r in batch]
    outputs = model(
        batch[0],
        attention_mask=batch[1]
    )
    # MLP layer
    logits = outputs.logits
    # logits = classfier(logits)

    #compare prediction and labels to get accuracy with batch
    prediction = torch.argmax(logits,dim=1)
    accuracy = torch.sum(prediction == batch[2]).item() / len(prediction)
    #count accuracy
    # release outputs on GPU
    del outputs
    del logits
    del batch
    total_accuracy += accuracy
print(total_accuracy/len(test_dataloader))



# 5. metrics for evaluation

