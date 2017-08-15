# -*- coding: utf-8 -*-

#Importing essential packages
from sklearn.feature_extraction.text import CountVectorizer
import glob
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#training function to import the data
def load_data(fname,label):
    configfiles = glob.glob(fname)
    labels=[]
    b=[]
    listdata=[]
    for files in configfiles:
        f=open(files)
        data=f.readlines()[15:]
        for i in data:
            k=re.sub(r'[^\w\s]','',i)
            k=k.lower()
            k=k.strip()
            listdata.append(k)
    b=['.'.join(listdata)]
    labels.append(label)
    return b,labels
    
#test function to import the data 
def load_data_test(fname,label):
    configfiles = glob.glob(fname)
    labels=[]
    a=[]
    listdata=[]
    for files in configfiles:
        listdata=[]
        f=open(files)
        data=f.readlines()[15:]
        for i in data:
            new_file=[]
            k=re.sub(r'[^\w\s]','',i)
            k=k.lower()
            k=k.strip()
            listdata.append(k)
        new_file=' '.join(listdata)
        a.append(new_file)
        labels.append(label)
    return a,labels
    
#Importing the training data      
comp_data, comp_lab =load_data('C:/python/training/comp/*','comp')
politics_data, politics_lab=load_data('C:/python/training/politics/*','politics')
rec_data, rec_lables=load_data('C:/python/training/rec/*','rec')
sports_data, sports_lab=load_data('C:/python/training/sports/*','sports')  

#Importing the test data
te_comp_data,te_comp_lab =load_data_test('C:/python/testing/comp/*','comp')
te_politics_data, te_politics_lab=load_data_test('C:/python/testing/politics/*','politics')
te_rec_data, te_rec_lab=load_data_test('C:/python/testing/rec/*','rec')
te_sports_data, te_sports_lab=load_data_test('C:/python/testing/sports/*','sports')  

#Combining train data 
train_data= comp_data+politics_data+rec_data+sports_data
train_lab= comp_lab+politics_lab+rec_lables+sports_lab

#Combining train data
test_data= te_comp_data+te_politics_data+te_rec_data+te_sports_data
test_lab= te_comp_lab+te_politics_lab+te_rec_lab+te_sports_lab  

#Build a counter based on the training dataset
counter=CountVectorizer()
counter.fit(train_data)

#count the number of times each term appears in a document and transform each doc into a count vector
cou_train=counter.transform(train_data) # for train_data
cou_test=counter.transform(test_data) #for test_data

#train classifier
clf=MultinomialNB(alpha=0.1)

#train all classifier on the same datasets
clf.fit(cou_train,train_lab)

#predicting the values for test data set
pred=clf.predict(cou_test)

#printing the accuracy
print(accuracy_score(pred,test_lab))


     

