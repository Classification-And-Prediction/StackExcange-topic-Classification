import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re, nltk        
from nltk.stem.porter import PorterStemmer
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

train_data_df = pd.read_csv('train_set.csv', header=None, delimiter="\t", quoting=3)
test_data_df = pd.read_csv('test_set.csv', header=None,delimiter="\n" , quoting=3 ,error_bad_lines=False)

train_data_df.columns = ["Domain","Text"]
test_data_df.columns = ["Text"]

print train_data_df.shape
print test_data_df.shape

print train_data_df.Domain.value_counts()

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("[http://]", " ", text)
    text = re.sub(" +"," ", text)
	text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text)
	text = re.sub("\\b[a-zA-Z0-9]{0,1}\\b"," ",text)
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 85)
 
corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())

corpus_data_features_nd = corpus_data_features.toarray()

corpus_data_features_nd.shape

vocab = vectorizer.get_feature_names()
print vocab

dist = np.sum(corpus_data_features_nd, axis=0)
for tag, count in zip(vocab, dist):
	print count, tag

X_train, X_test, y_train, y_test  = train_test_split(corpus_data_features_nd[0:len(train_data_df)], train_data_df.Domain, random_state=2) 

print "Logistic Regression MODEL \n"
logreg_model = LogisticRegression(penalty = 'l1',C = 0.6)
logreg_model = logreg_model.fit(X=X_train, y=y_train)
y_pred = logreg_model.predict(X_test)

accu_score = cross_val_score(logreg_model,X_test,y_test ,cv=10 ,scoring='accuracy').mean()
print "\n"
print "accuracy score : ",accu_score  
precision_score = cross_val_score(logreg_model,X_test,y_test,cv=10,scoring='precision').mean()
print "\n"
print "precision score : ",precision_score
recall_score = cross_val_score(logreg_model,X_test,y_test,cv=10,scoring='recall').mean()
print "\n"
print "recall score : ",recall_score
f1_score = cross_val_score(logreg_model,X_test,y_test,cv=10,scoring='f1').mean()
print "\n"
print "f1 score : ",f1_score,"\n"
