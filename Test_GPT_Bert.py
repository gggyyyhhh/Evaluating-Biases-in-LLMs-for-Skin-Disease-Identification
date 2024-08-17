import torch
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch.nn as nn
import pandas as pd
import tensorflow as tf

from Train_GPT_Bert import BertClassfication

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
#4.模型评估
test_set = pd.read_csv("seg_dataset/test/tem1.5_file_test.csv")#修改1
test_sentences = test_set.iloc[:, 2].values
test_targets = torch.tensor(test_set.iloc[:, 3].values).to(device)
hit = 0  # 初始化预测正确计数
total = len(test_sentences)  # 获取测试集总数
# 对测试集进行预测
bert_classfication=BertClassfication().to(device)
bert_classfication.load_state_dict(torch.load('weight//bert_tem1.5.pth'))#修改2
bert_classfication.eval()
with torch.no_grad():  # 确保评估过程中不计算梯度
    for i in range(total):
        outputs = bert_classfication([test_sentences[i]])
        _, predict = torch.max(outputs, 1)  # 获取预测结果
        print(predict, test_targets[i])
        if predict.item() == test_targets[i].item():  # 预测结果需要使用 .item() 获取标量  # 如果预测正确
            hit += 1
print('准确率为%.4f' % (hit / len(test_sentences)))  # 计算并输出准确率
