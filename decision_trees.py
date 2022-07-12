import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#  Importing the dataset
dataset = pd.read_csv("final_train1.csv")
x_train= dataset.text
y_train= dataset.sentiment

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=1)

dataset1 = pd.read_csv("final_test1.csv")
x_test1 = dataset1.text
y_test1 = dataset1.sentiment

#feature extraction
#cv = CountVectorizer()
#x_train_df = cv.fit_transform(x_train)
#print("dimensions of training data",x_train_df.shape)

#tf-idf transformer
#tfidf_transformer=TfidfTransformer()
#x_train_tf_idf=tfidf_transformer.fit_transform(x_train_df)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(x_train)
#using calssifier
#dt = DecisionTreeClassifier().fit(x_train_tf_idf, y_train)
dt = DecisionTreeClassifier().fit(vectors, y_train)
#x_test_df=cv.transform(x_test1)
#x_test2=tfidf_transformer.transform(x_test_df)
x_test2=vectorizer.transform(x_test1)


# Predicting a new result
y_test=dt.predict(x_test2)

#df = pd.DataFrame(y_test)
#df.to_csv (r'decision_tree_classifier.csv',header=['score'], index=None)


#printing resuts
from sklearn.metrics import recall_score, accuracy_score,f1_score

print ('Accuracy:', accuracy_score(y_test, y_test1))
print ('F1 score:', f1_score(y_test, y_test1,average='micro'))
print ('Recall:', recall_score(y_test, y_test1,average='micro'))
# 6 Visualising  results
#plt.plot(accuracy_score(y_test2, y_test1), , color = 'green')
# plt.title('Truth or Bluff (Support Vector Regression Model)')
# plt.xlabel('')
# plt.ylabel('Salary')
# plt.show()
