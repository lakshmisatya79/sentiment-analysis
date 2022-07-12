import pandas as pd

read_file = pd.read_csv (r'negative_test.txt')
read_file.to_csv (r'neg_test.csv',header=['text'], index=None)