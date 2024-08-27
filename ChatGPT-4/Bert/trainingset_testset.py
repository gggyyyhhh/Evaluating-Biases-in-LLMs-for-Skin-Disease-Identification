import os
import pandas as pd
from sklearn.model_selection import train_test_split

random_seed = 1

folder_path = 'API_result/label_result'
train_folder_path='Dataset/BERT/train'
test_folder_path='Dataset/BERT/test'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
for csv_file in csv_files:
    ori_file_path = os.path.join(folder_path, csv_file)
    try:
        data = pd.read_csv(ori_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(ori_file_path, encoding='ISO-8859-1')
        except Exception as e:
            print(f"error '{csv_file}': {e}")
            continue
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    base_name = os.path.splitext(csv_file)[0]
    train_file_path = os.path.join(train_folder_path, f'{base_name}_train.csv')
    test_file_path = os.path.join(test_folder_path, f'{base_name}_test.csv')
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)


