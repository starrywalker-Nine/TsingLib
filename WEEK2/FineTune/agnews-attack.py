# AG NEWS attack
import torch
import torch.nn as nn
import math
import numpy as np
import transformers
import time
from torch.utils.data import TensorDataset,Dataset,DataLoader 
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForTokenClassification, AdamW, BertConfig
from transformers import BertForSequenceClassification
import json
import torchtext
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenize(dataset):
    input_ids = []
    attention_masks = []
    for data in dataset:
        encoded_pair = tokenizer.encode_plus(
                data[1],                       # 句子1
                None,                    # 句子2
                max_length=128,            # 设置最大长度
                truncation=True,               # 超过最大长度则截断
                padding='max_length',          # 不足最大长度则填充
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        input_ids.append(encoded_pair['input_ids'])
        attention_masks.append(encoded_pair['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids,attention_masks

def flat_accuracy(logits,labels):
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



if __name__ == "__main__":
    # 1.1 download and load AG News dataset
    DATA_SIZE = 120000
    BATCH_SIZE = 128
    fine_tune_epochs = 10
    patience = 3
    best_acc = -1
    train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='.data', split=('train', 'test'))
    # 1.2 tokenize AG News dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids,attention_masks = tokenize(train_dataset)
    labels = torch.tensor([int(data[0])-1 for data in train_dataset])
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # tset dataset prepare
    test_input_ids,test_attention_masks = tokenize(test_dataset)
    test_labels = torch.tensor([int(data[0])-1 for data in test_dataset])
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # 1.3 load AG News model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    # 1.4 fine-tune AG News model
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.to(device)

    total_iterations = math.ceil(DATA_SIZE/BATCH_SIZE) * fine_tune_epochs
    progress_bar = tqdm(total=total_iterations, ncols=80)
    # training using tqdm
    for epoch in range(fine_tune_epochs):
        model.train()
        for batch in train_dataloader:
            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        # eval fine-tuned AG News model
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        for test_batch in test_dataloader:
            test_batch = [t.to(device) for t in test_batch]
            t_input_ids, t_attention_mask, t_labels = test_batch
            with torch.no_grad():
                outputs = model(t_input_ids, attention_mask=t_attention_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = t_labels.detach().cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy = (tmp_eval_accuracy+eval_accuracy)/2
        print(f'AG classification:accuracy{eval_accuracy}')
        if eval_accuracy > best_acc:
            best_acc = eval_accuracy
            patience =3  #early stopping
        else:
            patience -= 1
            if patience == 0:
                break
    progress_bar.close()
    # save fine-tuned AG News model
    torch.save(model.state_dict(), 'agnews-model.pt')


    

