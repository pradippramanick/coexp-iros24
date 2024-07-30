
import os
import torch
torch.manual_seed(0)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report


train_df = pd.read_csv("data/preprocessed/train.csv", delimiter='\t')
val_df = pd.read_csv("data/preprocessed/val.csv", delimiter='\t')
test_df = pd.read_csv("data/preprocessed/test.csv", delimiter='\t')

model_checkpoint = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)


print(val_df.columns)

label2id = dict(E=0, N=1, C=2)
id2label = list(label2id.keys())

train_premises = train_df.premise.values.tolist()
train_hyps = train_df.hypothesis.values.tolist()

assert len(train_premises) == len(train_hyps)

train_texts = [(p, h) for p, h in zip(train_premises, train_hyps)]

val_premises = val_df.premise.values.tolist()
val_hyps = val_df.hypothesis.values.tolist()

assert len(val_premises) == len(val_hyps)

val_texts = [(p, h) for p, h in zip(val_premises, val_hyps)]

train_labels = train_df.label.values.tolist()
val_labels = val_df.label.values.tolist()

train_encodings = tokenizer(train_texts, truncation=False, padding=True)
val_encodings = tokenizer(val_texts, truncation=False, padding=True)


class CADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(label2id[self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


print("TRAIN Dataset: {}".format(train_df.shape))
print("TEST Dataset: {}".format(val_df.shape))

train_dataset = CADataset(train_encodings, train_labels)
val_dataset = CADataset(val_encodings, val_labels)

epochs = 3
batch_size = 8
lr = 0.00005
ls = 0.05


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, predictions, average='macro')}


model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=3)
training_args = TrainingArguments(
    output_dir='./results',  
    num_train_epochs=epochs, 
    per_device_train_batch_size=batch_size,  
    per_device_eval_batch_size=batch_size,  
    warmup_steps=0, 
    weight_decay=0.01, 
    logging_dir='./logs',
    logging_steps=2,
    evaluation_strategy="epoch",
    learning_rate=lr, label_smoothing_factor=ls,
    dataloader_num_workers=1,
    dataloader_prefetch_factor=1,
    save_strategy='epoch',
    save_only_model=True,
    save_total_limit=2,
    metric_for_best_model='f1_macro',greater_is_better=True,
    load_best_model_at_end=True
)
trainer = Trainer(
    model=model,  
    args=training_args,  
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics
)


trainer.train()

test_premises = test_df.premise.values.tolist()
test_hyps = test_df.hypothesis.values.tolist()

test_texts = [(p, h) for p, h in zip(test_premises, test_hyps)]
test_dataset = CADataset(val_encodings, val_labels)

test_encodings = tokenizer(test_texts, truncation=False, padding=True)
test_labels = test_df.label.values.tolist()

test_dataset = CADataset(test_encodings, test_labels)

predictions = trainer.predict(test_dataset)

preds = np.argmax(predictions.predictions, axis=-1)
print(classification_report(predictions.label_ids, preds))

pt_save_directory = "DeBERTa-v3-base_fine_tuned" + str(epochs) + '_b' + str(batch_size) + '_lr' + str(lr) + '_ls' + str(ls)
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory, safe_serialization=False)