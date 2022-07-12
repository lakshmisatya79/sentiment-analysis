import pandas as pd

read_file = pd.read_csv (r'positive_test.txt')
read_file.to_csv (r'pos_test.csv',header=['text'], index=None)