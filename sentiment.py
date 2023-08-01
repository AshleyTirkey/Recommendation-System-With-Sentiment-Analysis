# Libraries

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn import over_sampling
from IPython.display import display

# Reading Data

df_prod_review = pd.read_csv('D:/college/Projects/PyCham Projects/Sentiment analysis/product_review_sentiment.csv',
                             encoding='latin-1')
display(df_prod_review.sample(n=5, random_state=42))

# Data Prep

x = df_prod_review['Review']
y = df_prod_review['user_sentiment']
print("Checking distribution of +ve and -ve review sentiment: \n{}".format(y.value_counts(normalize=True)))
# split data into test and train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
# balancing data using oversampling
ros = over_sampling.RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))
print("Checking distribution of +ve and -ve review sentiment after oversampling: \n{}".format(
    y_train.value_counts(normalize=True)))
X_train = X_train['Review'].tolist()

# Feature Eng(converting into numbers)

word_vetorizer = TfidfVectorizer(strip_accents='unicode', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                 stop_words='english', sublinear_tf=True)
# fitting on train
word_vetorizer.fit(X_train)
# transforming train and test datasets
X_train_transformed = word_vetorizer.transform(X_train)
X_test_transformed = word_vetorizer.transform(X_test.tolist())


# Building the Model (Logistic Regression)

def evaluate_model(y_pred, y_actual):
    print(classification_report(y_true=y_actual, y_pred=y_pred))
    # confusion matrix
    cm = confusion_matrix(y_true=y_actual, y_pred=y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    # calculating sensitivity
    sensitivity = round(TP / float(FN + TP), 2)
    print("sensitivity: {}".format(sensitivity))
    # calculating the specificity
    specificity = round(TN / float(TN + FP), 2)
    print("specificity: {}".format(specificity))


# model training
logit = LogisticRegression()
logit.fit(X_train_transformed, y_train)
# prediction on train data
y_pred_train = logit.predict(X_train_transformed)
# prediction on test data
y_pred_test = logit.predict(X_test_transformed)
# evaluation on train
print("Evaluation on Train Dataset..")
evaluate_model(y_pred=y_pred_train, y_actual=y_train)
# evaluation on test
print("Evaluation on Test dataset..")
evaluate_model(y_pred=y_pred_test, y_actual=y_test)

# Saving Model
filename = 'logit_model.pkl'
pickle.dump(logit,open(filename, 'wb'))
filename_other = 'word_vectorizer.pkl'
pickle.dump(word_vetorizer,open(filename_other, 'wb'))

