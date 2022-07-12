import pandas as pd
import io
import csv


def balancing(count):
    x = []
    if count >= 2:
        x = +2
    elif 0 < count < 2:
        x = +1
    elif -2 < count < 0:
        x = -1
    elif count <= -2:
        x = -2
    elif count == 0:
        x = 0
    #    count = 0
    return x
df = pd.read_csv('final_train.csv')
s = df.text
s1 = df.score

with io.open('final_train1.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['text','score', 'sentiment'])
    for (line, d) in zip(s, s1):
        writer.writerow([line,d, balancing(d)])