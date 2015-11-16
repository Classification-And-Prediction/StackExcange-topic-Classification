import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re, nltk        
from nltk.stem.porter import PorterStemmer
import random
import matplotlib.pyplot as plt  
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

#Validation Curve Using Naive Bayes

l = [0.01,1.00,100.00]
for i in range(0,3) :
	
	param_range = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	train_scores, test_scores = validation_curve(MultinomialNB(fit_prior = False), X_test, y_test,param_name="alpha",param_range=param_range,cv=10)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.title("Validation Curve with NB")
	plt.xlabel("$\gamma$")
	plt.ylabel("Score")
	plt.ylim(0.0, 1.1)
	plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
	             color="g")
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
	plt.legend(loc="best")
	
	plt.show()
