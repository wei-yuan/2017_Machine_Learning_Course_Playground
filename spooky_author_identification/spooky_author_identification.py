# --------------------------------------
# Library
# --------------------------------------
# Tools
import codecs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import nltk
import csv
import unicodecsv as csv

# Data preprocessor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

# Model
from sklearn.neural_network import MLPClassifier

# Validation tool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Read data set
def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

data = pd.read_csv("data_set/train.csv")
# convert csv to unicode
utf_8_encoder(data)
unicode_csv_reader(data)
train = data[['text']]
label = data[['author']]

print data.head()

# check data balance
print("Number of rows in train dataset : ",data.shape[0])
'''
cnt_author = data['author'].value_counts()

plt.figure(figsize=(10,4))
sns.barplot(cnt_author.index, cnt_author.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Author Name', fontsize=12)
plt.show()
'''
# --------------------------------------
# NLP basic step
# --------------------------------------
# Tokenization
#tokens = nltk.word_tokenize(str(train))
tokens = data['text'].apply(word_tokenize)
tokens = train.apply(word_tokenize)
print tokens

# Filtering: Remove stopwords of English
stopwords = nltk.corpus.stopwords.words('english')

text_without_stopwords = [word for word in tokens if word.lower() not in stopwords]
print(text_without_stopwords)
print("="*90)
print("Length of original list: {0} words\n"
      "Length of list after stopwords removal: {1} words"
      .format(len(tokens), len(text_without_stopwords)))

# Normalization

# Stemming and Lemmatization
stemmer = nltk.stem.PorterStemmer()
lemma = WordNetLemmatizer()

# Word Encoding: Vectorization of word
vectorizer = CountVectorizer(min_df=0)
text_without_stopwords_transform = vectorizer.fit_transform(text_without_stopwords)

print("The features are:\n {}".format(vectorizer.get_feature_names()))
print("\nThe vectorized array looks like:\n {}".format(text_without_stopwords_transform.toarray()))
