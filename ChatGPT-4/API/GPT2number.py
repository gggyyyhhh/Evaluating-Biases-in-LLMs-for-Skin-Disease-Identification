import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("Dataset/BERT/test/tem0_file_test.csv")
def convert_label(value):
    original_value = value
    value = value.lower()
    if ('melanoma' in value and 'nevus' in value) or \
       ('melanoma' in value and 'nevi' in value) or \
       ('melanoma' in value and 'keratosis' in value) or \
       ('nevus' in value and 'keratosis' in value) or \
       ('nevi' in value and 'keratosis' in value):
        result=3
    elif 'melanoma' in value or 'skin cancer' in value:
        result = 0
    elif 'nevi' in value or 'nevus' in value:
        result = 1
    elif 'keratosis' in value:
        result = 2
    else:
        result = 3
    print(f"Original: {original_value} | Converted: {result}")
    return result

df['converted_column'] = df['Response2'].apply(convert_label)
true_labels = df.iloc[:, 4].values
accuracy = accuracy_score(true_labels, df['converted_column'])
print("Accuarcy: {:.2f}%".format(accuracy * 100))