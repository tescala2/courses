#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class DF_2_Feature_S(object):
    # converts from df to list of features separated by sentence
    def __init__(self, text):
        self.num = 1
        func = lambda f: [(word, pos) for word, pos in zip(f['Word'].values.tolist(), f['POS'].values.tolist())]
        self.group = text.groupby('Sentence #').apply(func)
        self.sents = [f for f in self.group]

    def get_next(self):
        f = self.group['Sentence: {}'.format(self.num)]
        self.num += 1
        return f
        
class DF_2_Target_S(object):
    # converts from df to list of labels separated by sentence
    def __init__(self, text):
        self.num = 1
        func = lambda l: [(tag) for tag in l['Tag'].values.tolist()]
        self.group = text.groupby('Sentence #').apply(func)
        self.sents = [l for l in self.group]

    def get_next(self):
        l = self.group['Sentence: {}'.format(self.num)]
        self.num += 1
        return l

class FF(nn.Module):
    def __init__(self, num_words, emb_dim, num_y):
        super().__init__()
        self.emb = nn.Embedding(num_words, emb_dim)
        self.linear = nn.Linear(emb_dim, num_y)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        embeds = self.emb(text)
        return self.softmax(self.linear(embeds))

def load_vocab(text):
    feat_to_idx = {}
    for sent in text:
        for feat in sent:
            feat_to_idx.setdefault(feat, len(feat_to_idx))
    return feat_to_idx

def load_tag(text):
    tag_to_idx = {}
    for sent in text:
        for tag in sent:
            tag_to_idx.setdefault(tag, len(tag_to_idx))
    return tag_to_idx

# read in the data
train_data = pd.read_csv('trainDataWithPOS.csv')
test_data = pd.read_csv('testDatawithPOS.csv')

# preprocess data
train_sent_feature = DF_2_Feature_S(train_data).sents
train_sent_label = DF_2_Target_S(train_data).sents
x_test = DF_2_Feature_S(test_data).sents
y_test = DF_2_Target_S(test_data).sents

# split the data into 80% train and 20% validation
x_train, x_val, y_train, y_val = train_test_split(train_sent_feature, train_sent_label, test_size=0.2)

print('The features used are the word and the POS')

# convert features and labels to numbers
feat_to_idx = load_vocab(x_train)
feat_to_idx[('UNK')] = len(feat_to_idx)-1
tag_to_idx = load_tag(y_train)

# changes the features from tuples to strings
x_train_ = []
x_val_ = []
x_test_ = []
old_x_data = [x_train, x_val, x_test]
new_x_data = [x_train_, x_val_, x_test_]

for i in range(len(old_x_data)):
    for sentence in old_x_data[i]:
        new_sent = []
        for feat in sentence:
            new_x_data[i].append(str(feat))

# changes the tags from tuples to strings of indices
y_train_ = []
y_val_ = []
y_test_ = []
old_y_data = [y_train, y_val, y_test]
new_y_data = [y_train_, y_val_, y_test_]

for i in range(len(old_y_data)):
    for sentence in old_y_data[i]:
        new_sent = []
        for tag in sentence:
            tag_idx = tag_to_idx[tag]
            new_y_data[i].append(str(tag_idx))

cv = CountVectorizer()
train_data = cv.fit_transform(x_train_)
val_data = cv.transform(x_val_)
test_data = cv.transform(x_test_)

MNB = MultinomialNB()
MNB.fit(train_data, np.array(y_train_).flatten())
MNB_train_acc = MNB.score(train_data, np.array(y_train_).flatten())
MNB_val_acc = MNB.score(val_data, np.array(y_val_).flatten())
MNB_test_acc = MNB.score(test_data, np.array(y_test_).flatten())
print('Naive Bayes Training Accuracy:', MNB_train_acc)
print('Naive Bayes Validation Accuracy:', MNB_val_acc)
print('Naive Bayes Test Accuracy:', MNB_test_acc)

LR = LogisticRegression(max_iter=1500)
LR.fit(train_data, np.array(y_train_).flatten())
LR_train_acc = LR.score(train_data, np.array(y_train_).flatten())
LR_val_acc = LR.score(val_data, np.array(y_val_).flatten())
LR_test_acc = LR.score(test_data, np.array(y_test_).flatten())
print('Logistic Regression Train Accuracy:', LR_train_acc)
print('Logistic Regression Validation Accuracy:', LR_val_acc)
print('Logistic Regression Test Accuracy:', LR_test_acc)

preds = LR.predict(test_data)
report = classification_report(np.array(y_test_).flatten(), preds)
print('\n')
print(report)

emb_dim = 30
learning_rate = 0.5
model = FF(len(feat_to_idx), emb_dim, len(tag_to_idx))
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
print('Feed Forward Network')
print('Hyperparameters: emb_dim = 30, lr = 0.5, optim = SGD, loss = NLLL, epochs = 10')

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for i in range(len(x_train)):
        x = [feat_to_idx[feat] for feat in x_train[i]]
        y = [tag_to_idx[tag] for tag in y_train[i]]
        x_train_tensor = torch.LongTensor(x)
        y_train_tensor = torch.LongTensor(y)
        pred_y = model(x_train_tensor)
        loss = loss_fn(pred_y, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("\nEpoch:", epoch)
    print("Training loss:", loss.item())

with torch.no_grad():
    model.eval()
    correct = 0
    for i in range(len(x_val)):
        x = [feat_to_idx.get(feat, len(feat_to_idx)-1) for feat in x_val[i]]
        y = [tag_to_idx[tag] for tag in y_val[i]]
        x_val_tensor = torch.LongTensor(x)
        y_val_tensor = torch.LongTensor(y)
        pred_y_val = model(x_val_tensor)
        pred_y_val = torch.argmax(pred_y_val, dim=1)
        num_correct = 0
        for j in range(len(pred_y_val)):
            num_correct += (pred_y_val[j] == y_val_tensor[j]).sum().item()
        correct += num_correct
    print('Feed Forward Network Validation Accuracy:', correct/len(x_val_))

with torch.no_grad():
    model.eval()
    correct = 0
    for i in range(len(x_test)):
        x = [feat_to_idx.get(feat, len(feat_to_idx)-1) for feat in x_test[i]]
        y = [tag_to_idx[tag] for tag in y_test[i]]
        x_test_tensor = torch.LongTensor(x)
        y_test_tensor = torch.LongTensor(y)
        pred_y_test = model(x_test_tensor)
        pred_y_test = torch.argmax(pred_y_test, dim=1)
        num_correct = 0
        for j in range(len(pred_y_test)):
            num_correct += (pred_y_test[j] == y_test_tensor[j]).sum().item()
        correct += num_correct
    print('Feed Forward Network Test Accuracy:', correct/len(x_test_))

print('results not far off from naive bayes and logistic regression')

