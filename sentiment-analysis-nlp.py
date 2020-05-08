"""This is "sentiment-analysis-for-financial-news" dataset from Kaggle. This dataset contains the sentiments for financial news headlines from the perspective of a retail investor. The dataset contains two columns, "Sentiment" and "News Headline". The sentiment can be negative, neutral or positive"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('sentiment-analysis-for-financial-news.csv')
X_text = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values
y = y.reshape(-1, 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 4845):
    review = re.sub('[^a-zA-Z]', ' ', X_text[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#-------Applying Random Forest Classifier on the data------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((cm[0, 0]+cm[1, 1]+cm[2, 2])/
                       (cm[0, 0]+cm[1, 1]+cm[2, 2]+cm[0, 1]+cm[1, 0]+cm[0, 2]+cm[2, 0]+cm[1, 2]+cm[2, 1]))*100


#----k-fold cross validation of the training set-----------------------------
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean_accuracy_of_training_set = accuracies.mean()*100
std_of_training_set = accuracies.std()


