#  -*- coding: utf-8 -*-

import json
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score

def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels) 
    return (preds == labels).mean()
  
def pre_recall_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    # recall=TP/(TP+FN)
    tp = np.sum((labels == '1') & (preds == '1'))
    fp = np.sum((labels == '0') & (preds == '1'))
    fn = np.sum((labels == '1') & (preds == '0'))
    r = tp * 1.0 / (tp + fn)
    # Precision=TP/(TP+FP)
    p = tp * 1.0 / (tp + fp)
    epsilon = 1e-31
    f1 = 2 * p * r / (p+r+epsilon)
    return p, r, f1

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def pre_recall_f1_(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = simple_accuracy(preds, labels)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
            "acc": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            # "acc_and_f1": (acc + f1) / 2,
        }

def res_evaluate(res_dir="./outputs/predict_classification_type/predictions.json", eval_phase='test'):
    if eval_phase == 'test':
        data_dir="./data/dialog/type/dev.tsv"
    elif eval_phase == 'dev':
        data_dir="./data/dev.tsv"

    else:
        assert eval_phase in ['dev', 'test'], 'eval_phase should be dev or test'
    
    labels = []
    with open(data_dir, "r") as file:
        first_flag = True
        for line in file:
            line = line.split("\t")
            label = line[0]
            if label=='label':
                continue
            labels.append(str(label))
    file.close()

    preds = []
    with open(res_dir, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            pred = line['label']
            preds.append(str(pred))
    file.close()
    print(len(labels))
    print(len(preds))
    assert len(labels) == len(preds), "prediction result doesn't match to labels"
    print('data num: {}'.format(len(labels)))
    #p, r, f1 = pre_recall_f1(preds, labels)
    #print("accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(accuracy(preds, labels), p, r, f1))
    print(pre_recall_f1_(preds, labels))
res_evaluate()
