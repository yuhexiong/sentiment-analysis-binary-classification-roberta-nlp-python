# -*- coding: utf-8 -*-

!pip install transformers datasets wandb

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from transformers import Trainer
from transformers import TrainingArguments, EarlyStoppingCallback, IntervalStrategy

import numpy as np
from datasets import load_metric
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import torch

# read train, valid, test csv
# all csv column: id, title, comment, label(0, 1)
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')
test = pd.read_csv('test.csv')


train = train.drop(columns=['id', 'title'])
valid = valid.drop(columns=['id', 'title'])
test = test.drop(columns=['id', 'title'])

# remove comment longer than 500
train = train[train['comment'].str.len() < 500]
valid = valid[valid['comment'].str.len() < 500]
test = test[test['comment'].str.len() < 500]

# convert pandas as dataset format, combine as datasetDict format
train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)
validation_dataset = Dataset.from_pandas(valid)
my_dataset_dict = DatasetDict({"train":train_dataset,"validation":validation_dataset ,"test":test_dataset})

# choose tokenizer 
tokenizer_name = "uer/roberta-base-finetuned-chinanews-chinese"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def tokenize(batch):
    return tokenizer(batch["comment"], padding=True, truncation=True)


my_dataset_dict_encoded = my_dataset_dict.map(tokenize, batched=True, batch_size=None)
my_dataset_dict_encoded = my_dataset_dict_encoded.remove_columns('comment')
my_dataset_dict_encoded = my_dataset_dict_encoded.remove_columns('__index_level_0__')

# setting device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (AutoModelForSequenceClassification
        .from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese', num_labels=2
        ,id2label={"0": "negative",
            "1": "positive"}
        ,label2id={"negative": "0",
            "positive": "1"})
        .to(device))

# setting parameter
batch_size = 8
logging_steps = len(my_dataset_dict_encoded["train"]) // batch_size
model_name = "my_model"
labels = ["negative", "positive"]

training_args = TrainingArguments(output_dir=model_name,
                num_train_epochs=25,
                learning_rate=1e-4,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                disable_tqdm=False,
                label_names= labels,
                report_to = "wandb",
                logging_steps=logging_steps)

# define accuracy
def compute_metrics(pred):
    labels = pred.label_id
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# train model
trainer = Trainer(model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=my_dataset_dict_encoded["train"],
        eval_dataset=my_dataset_dict_encoded["validation"],
        tokenizer=tokenizer)
trainer.train()

# save model
trainer.save_model('robertaModel')

# classify testing data and save as csv
test = pd.read_csv('test.csv')

classifier = pipeline(task= 'sentiment-analysis', model= "robertaModel")
test_model_result = classifier(test['comment'].to_list())

for i in range(len(test)):
    test.loc[i, 'prediction'] = int(test_model_result[i]['label'][5])

test.to_csv('test_prediction.csv', encoding='utf-8-sig')