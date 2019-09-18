# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 23:38:22 2019

@author: Sriharsha Komera
"""

### Imprting my dataset
import pandas as pd

path= 'F:\\Krish\\NLP\\Spam Classifier\\SpamClassifier-master\\smsspamcollection\\SMSSpamCollection'

messages=pd.read_csv(path, sep='\t', names=["label","message"])

### Data Cleaning and pre-processing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
ps=PorterStemmer()
lm=WordNetLemmatizer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-z]',' ',messages['message'] [i])
    review=review.lower()
    review=review.split()
    
    ##review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=[lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

###Creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

###train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=0)

###Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)

y_pred= spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
