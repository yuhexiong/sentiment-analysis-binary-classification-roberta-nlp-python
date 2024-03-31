# Sentiment Analysis

Using the Hugging Face's RoBERTa model for fine-tuning to achieve the desired Chinese sentiment analysis.  
Preparation involves organizing all crawled data into fields containing id, title (if unavailable, leave blank as it's not the target for this training), comment, and label (expressed as 0 or 1 for negative or positive, respectively). An adequate amount of data is selected to label the sentiment of comment in the label field. The labeled data is then split into training and validation sets according to the desired ratio, while unlabeled data is designated for testing. Consequently, three CSV files are prepared: train.csv, valid.csv, and test.csv, enabling training to commence.  
I utilize Google's Colab to run my model, allowing for hardware-specific training adjustments. In the code, the model is stored as 'robertaModel'. Once training is completed, the upper portion of the code can be commented out, and the trained model can be used for analyzing different testing datasets iteratively.  

## Overview

- Language: Python v3.10.12
- Model: RoBERTa(Hugging Face)

## Run

```
python sentiment-analysis-roberta-nlp.py
```


## Example CSV To Be Prepared

### train.csv

| id | title                | comment                                      | label |
|----|----------------------|----------------------------------------------|-------|
| 1  | 軟體工程師offer請益   | 果斷選第一個offer 超優                        | 1     |
| 2  | 軟體工程師offer請益   | 軟體工程師的薪水的起薪真慘                     | 0     |
| 3  | 軟體工程師offer請益   | 都很差，再繼續投                              | 0     |
| 4  | 軟體工程師offer請益   | 非本科這樣算不錯了！                          | 1     |

### valid.csv

| id | title                | comment                                      | label |
|----|----------------------|----------------------------------------------|-------|
| 5  | 軟體工程師職業生涯     | 上班壓力很大                                  | 0     |
| 6  | 軟體工程師職業生涯     | 我自己是覺得滿有趣的                           | 1     |
| 7  | 軟體工程師職業生涯     | 感覺很容易會被 AI 取代                        | 0     |
| 8  | 軟體工程師職業生涯     | 好憂慮，競爭太激烈了                          | 0     |

### test.csv

| id | title                | comment                                      | label |
|----|----------------------|----------------------------------------------|-------|
| 9  | 軟體工程師的技能要求   | 軟體算好上手的了                              | 1     |
| 10 | 軟體工程師的技能要求   | 看天分，學不會就是學不會，邏輯很難              | 0     |
| 11 | 軟體工程師的技能要求   | 而且要持續學習，所以很辛苦                     | 0     |
| 12 | 軟體工程師的技能要求   | 缺乏最新的技術知識可能會影響我的工作機會。       | 0     |
