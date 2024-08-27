import torch
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch.nn as nn
import pandas as pd
import tensorflow as tf

from Train_GPT_Bert import BertClassfication

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
test_set = pd.read_csv("Dataset/BERT/test/tem0_file_test.csv")
test_sentences = test_set.iloc[:, 2].values
test_targets = torch.tensor(test_set.iloc[:, 3].values).to(device)
hit = 0
total = len(test_sentences)
bert_classfication=BertClassfication().to(device)
bert_classfication.load_state_dict(torch.load('weights/BERT/bert_tem0.pth'))
bert_classfication.eval()
with torch.no_grad():
    for i in range(total):
        outputs = bert_classfication([test_sentences[i]])
        _, predict = torch.max(outputs, 1)
        print(predict, test_targets[i])
        if predict.item() == test_targets[i].item():
            hit += 1
print('Accuarcy:%.4f' % (hit / len(test_sentences)))
