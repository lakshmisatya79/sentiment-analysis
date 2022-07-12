import pandas as pd

read_file = pd.read_csv (r'negative_train.txt')
read_file.to_csv (r'neg_train.csv',header=['text'], index=None)