import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

from sklearn import metrics
# Predicted values
# y_pred = ["a", "b", "c", "a", "b"]
# Actual values
# y_act = ["a", "b", "c", "c", "a"]
# y_pred = [0,1,0,1,1,1]
# y_act = [0,1,1,0,0,1]
# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
# print(metrics.confusion_matrix(y_act, y_pred, labels=[0, 1]))
# Printing the precision and recall, among other metrics
# print(metrics.classification_report(y_act, y_pred, labels=[0, 1]))

np.random.seed(500)

# res = [0,1,0,1,0,1]
# target = [0,1,0,1,0,1]

# precision = 
# recall = 


# Corpus = pd.read_csv(r"C:\Users\gunjit.bedi\Desktop\NLP Project\corpus.csv",encoding='latin-1')
# c1 = pd.read_csv(r"/Users/yangzhang/Downloads/jigsawCorpusNormal/train.csv")

c1 = pd.read_csv(r"out.csv")

Corpus = c1

text = Corpus["comment_text"]
labels = Corpus["Toxic-or-not"]


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], labels,test_size=0.2)

# print(Train_X)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


Tfidf_vect = TfidfVectorizer(max_features=500)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Tfidf_vect.vocabulary_)

print(Train_X_Tfidf)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

print(metrics.confusion_matrix(Test_Y, predictions_NB, labels=[0, 1]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(Test_Y, predictions_NB, labels=[0, 1]))

