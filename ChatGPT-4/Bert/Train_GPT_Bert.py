import random
import torch
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch.nn as nn
import pandas as pd
import tensorflow as tf
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 1
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication, self).__init__()
        self.model_path = 'bert_base_uncased/'
        self.model = BertModel.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.fc = nn.Linear(768, 4)
    def forward(self, x):
        batch_tokenized = self.tokenizer.batch_encode_plus(
            x,
            add_special_tokens=True,
            max_length=157,
            pad_to_max_length=True
            )
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
        hiden_outputs = self.model(input_ids, attention_mask=attention_mask)
        outputs = hiden_outputs[1]
        output = self.fc(outputs)
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = pd.read_csv("Dataset/BERT/train/tem0_file_train.csv")
    sentences = train_set.iloc[:, 2].values
    targets = train_set.iloc[:, 3].values
    batch_size = 32
    batch_count = int(len(sentences) / batch_size)
    batch_train_inputs, batch_train_targets = [], []
    for i in range(batch_count):
        batch_train_inputs.append(sentences[i * batch_size: (i + 1) * batch_size])
        batch_train_targets.append(targets[i * batch_size: (i + 1) * batch_size])

    bertclassfication = BertClassfication().to(device)
    bertclassfication.train()
    lossfuction = nn.CrossEntropyLoss()
    optimizer = AdamW(bertclassfication.parameters(), lr=2e-5)
    epoch = 10
    for _ in range(epoch):
        los = 0
        for i in range(batch_count):
            inputs = batch_train_inputs[i]
            targets = torch.tensor(batch_train_targets[i]).to(device)
            optimizer.zero_grad()
            outputs = bertclassfication(inputs)
            loss = lossfuction(outputs, targets)
            loss.backward()
            optimizer.step()
            los += loss.item()
            if i % 5 == 0:
                print("Batch:%d, Loss %.4f" % (i, los / 5))
                los = 0

    torch.save(bertclassfication.state_dict(), 'weights/BERT/bert_tem0.pth')
