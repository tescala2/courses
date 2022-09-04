#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import nltk


# ## Using the movie reviews data provided (and already tokenized) for you, you will make a new Python 3 sentiment_classifier.py file where you will implement Naive Bayes and logistic regression for predicting the sentiment of movie reviews in the validation data.

# In[2]:


train_sents = []
train_labels = []
val_sents = []
val_labels = []

with open('train.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        train_sents.append(line[1:])
        train_labels.append(int(line[0]))

with open('val.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        val_sents.append(line[1:])
        val_labels.append(int(line[0]))
        
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)


# ### First, as a simple baseline, implement a classifier that always predicts label 1 (i.e. positive), and evaluate it on both the training and validation datasets. You should see an accuracy of roughly 50%, which is just as bad as a random guess!

# In[3]:


ones_pred_train = np.ones(np.shape(train_labels))
ones_pred_val = np.ones(np.shape(val_labels))

train_scores = np.where(ones_pred_train == train_labels, 1.0, 0.0)
train_acc = np.mean(train_scores)

val_scores = np.where(ones_pred_val == val_labels, 1.0, 0.0)
val_acc = np.mean(val_scores)

print('Random Train Accuracy:', train_acc)
print('Random Validation Accuracy:', val_acc)


# ### Implement a multinomial Naive Bayes classifier with the scikit-learn toolkit that uses unigram feaures. 
# ### What are your training and validation accuracies?

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


train = [' '.join(ele) for ele in train_sents]
val = [' '.join(ele) for ele in val_sents]


# In[7]:


cv = CountVectorizer()
train_data = cv.fit_transform(train)
val_data = cv.transform(val)


# In[8]:


from sklearn.naive_bayes import MultinomialNB


# In[9]:


MNB = MultinomialNB()
MNB.fit(train_data, train_labels)
MNB_train_acc = MNB.score(train_data, train_labels)
print('Naive Bayes Training Accuracy:', MNB_train_acc)


# In[10]:


MNB_preds = MNB.predict(val_data)
score = np.where(MNB_preds == val_labels, 1.0, 0.0)
mean = np.mean(score)
print('Naive Bayes Validation Accuracy:', mean)


# ### Now build a logistic regression classifier, again using the scikit-learn toolkit with unigram features.
# ### How does it compare to Naive Bayes?

# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


LR = LogisticRegression(max_iter=1500)
LR.fit(train_data, train_labels)
LR_train_acc = LR.score(train_data, train_labels)
print('Logistic Regression Train Accuracy:', LR_train_acc)


# In[13]:


LR_preds = LR.predict(val_data)
score = np.where(LR_preds == val_labels, 1.0, 0.0)
mean = np.mean(score)
print('Ligistic Regression Validation Accuracy:', mean)


# ### Finally, add bigram features to the logistic regression. 
# ### Do the new accuracies match your intuition?

# In[14]:


cv2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
train_data = cv2.fit_transform(train)
val_data = cv2.transform(val)


# In[15]:


LR2 = LogisticRegression(max_iter=1500)
LR2.fit(train_data, train_labels)
LR2_train_acc = LR2.score(train_data, train_labels)
print('Logistic Regression Train Accuracy:', LR2_train_acc)


# In[16]:


LR2_preds = LR2.predict(val_data)
score = np.where(LR2_preds == val_labels, 1.0, 0.0)
mean = np.mean(score)
print('Logistic Regression Validation Accuracy:', mean)

