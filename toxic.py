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
y_pred = [0,1,0,1,1,1]
y_act = [0,1,1,0,0,1]
# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
print(metrics.confusion_matrix(y_act, y_pred, labels=[0, 1]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_act, y_pred, labels=[0, 1]))

np.random.seed(500)

# res = [0,1,0,1,0,1]
# target = [0,1,0,1,0,1]

# precision = 
# recall = 


# Corpus = pd.read_csv(r"C:\Users\gunjit.bedi\Desktop\NLP Project\corpus.csv",encoding='latin-1')
# c1 = pd.read_csv(r"/Users/yangzhang/Downloads/jigsawCorpusNormal/train.csv")

c1 = pd.read_csv(r"/Users/yangzhang/Downloads/jigsawCorpusNormal/train-toxic-var.csv")

Corpus = c1[:1000]

text = Corpus["comment_text"]
labels = Corpus["Toxic-or-not"]

print(Corpus)

# Step - a : Remove blank rows if any.
text.dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
text = [entry.lower() for entry in text]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
text = [word_tokenize(entry) for entry in text]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
res = []
for index,entry in enumerate(text):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], labels,test_size=0.3)

print(Train_X)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


Tfidf_vect = TfidfVectorizer(max_features=5000)
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