import pandas as pd

read_file = pd.read_csv (r'positive_train.txt')
read_file.to_csv ('pos_train.csv',header=['text'], index=None)