import random
import torch
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch.nn as nn
import pandas as pd
import tensorflow as tf
import numpy as np

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
seed = 1
set_seed(seed)

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
#2.定义bert分类模型
class BertClassfication(nn.Module):  # 继承自nn.Module类
    def __init__(self):
        super(BertClassfication, self).__init__()
        self.model_path = 'bert_base_uncased/'
        self.model = BertModel.from_pretrained(self.model_path)  # 加载预训练的BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)  # 加载对应的分词器
        self.fc = nn.Linear(768, 4)  # 定义一个全连接层，输入维度为768(self.model.config.hidden_size)，输出维度为3

    def forward(self, x):  # 定义前向传播方法
        # 对输入的文本进行分词、编码和填充
        batch_tokenized = self.tokenizer.batch_encode_plus(#使用bert的分词器对输入文本x进行处理
            x,
            add_special_tokens=True,#添加特殊标记，例如 [CLS] 和 [SEP]
            max_length=157,
            pad_to_max_length=True
            )
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)  # 移动到 GPU  # 获取输入的编码
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)  # 获取注意力掩码
        hiden_outputs = self.model(input_ids, attention_mask=attention_mask)  # 输入到BERT模型中，获取隐藏层输出
        #outputs = hiden_outputs[0][:, 0, :]  # 获取[CLS]标记对应的结果
        outputs = hiden_outputs[1]
        output = self.fc(outputs)  # 输入到全连接层中，获取分类结果
        return output

if __name__ == "__main__":
    # 1.创建训练集
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_set = pd.read_csv("seg_dataset/train/tem1.5_file_train.csv")#修改1
    sentences = train_set.iloc[:, 2].values
    targets = train_set.iloc[:, 3].values
    batch_size = 32  # 设置批处理大小
    batch_count = int(len(sentences) / batch_size)  # 计算一共有多少批次
    batch_train_inputs, batch_train_targets = [], []  # 一个列表存储分段的文本，一个列表存储分段的标签
    for i in range(batch_count):
        batch_train_inputs.append(sentences[i * batch_size: (i + 1) * batch_size])
        batch_train_targets.append(targets[i * batch_size: (i + 1) * batch_size])

    # 3.模型训练
    # 实例化模型
    bertclassfication = BertClassfication().to(device)
    bertclassfication.train()
    # 定义损失函数
    lossfuction = nn.CrossEntropyLoss()  # 适用于多分类问题，计算包括内部的softmax
    # 定义优化器
    optimizer = AdamW(bertclassfication.parameters(), lr=2e-5)
    epoch = 10  # 设置训练轮数
    for _ in range(epoch):
        los = 0  # 初始化损失
        for i in range(batch_count):
            inputs = batch_train_inputs[i]
            targets = torch.tensor(batch_train_targets[i]).to(device)
            optimizer.zero_grad()  # 梯度置零
            outputs = bertclassfication(inputs)  # 前向传播，获取输出
            loss = lossfuction(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            los += loss.item()  # 累加损失
            # 每处理五个批次，输出一次损失
            if i % 5 == 0:
                print("Batch:%d, Loss %.4f" % (i, los / 5))
                los = 0

    # 保存模型权重
    torch.save(bertclassfication.state_dict(), 'weight//bert_tem1.5.pth')#修改2
    """
    #4.模型评估
    test_set = pd.read_csv("Dataset//test.csv", header=None)
    test_sentences = train_set.iloc[:, 1].values
    test_targets = torch.tensor(train_set.iloc[:, 0].values).to(device)
    hit = 0  # 初始化预测正确计数
    total = len(test_sentences)  # 获取测试集总数
    # 对测试集进行预测
    bertclassfication.eval()
    for i in range(total):
        outputs = bertclassfication([test_sentences[i]])
        _, predict = torch.max(outputs, 1)  # 获取预测结果
        print(predict,test_targets[i])
        if predict.item() == test_targets[i].item():  # 预测结果需要使用 .item() 获取标量  # 如果预测正确
            hit += 1
    print('准确率为%.4f' % (hit / len(test_sentences)))  # 计算并输出准确率
    """
