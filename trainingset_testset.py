import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 设置随机种子
random_seed = 1

# 设置文件夹路径
folder_path = 'all_dataset'
train_folder_path='GPT_seg_dataset/train'
test_folder_path='GPT_seg_dataset/test'
# 获取文件夹中所有 CSV 文件的列表
#csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files = ['tem1.5_response2_file.csv']
# 读取 CSV 文件
for csv_file in csv_files:
    ori_file_path = os.path.join(folder_path, csv_file)
    #data = pd.read_csv(ori_file_path)
    try:
        data = pd.read_csv(ori_file_path, encoding='utf-8')  # 尝试使用 UTF-8 编码
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(ori_file_path, encoding='ISO-8859-1')  # 尝试使用 ISO-8859-1 编码
        except Exception as e:
            print(f"无法读取文件 '{csv_file}': {e}")
            continue
    # 从数据集中随机抽取 80% 作为训练集，20% 作为测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    # 将训练集和测试集保存到新的 CSV 文件中
    base_name = os.path.splitext(csv_file)[0]  # 获取不带扩展名的文件名
    train_file_path = os.path.join(train_folder_path, f'{base_name}_train.csv')
    test_file_path = os.path.join(test_folder_path, f'{base_name}_test.csv')
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)


