#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.spatial.distance import pdist, squareform


# In[2]:


# Q1 Load in the data using the pandas read_csv function.

data=pd.read_csv('tunedit_genres.csv')


# In[3]:


#The last variable 'RockOrNot' determines whether the music genre for that sample is rock or not
# What percentage of the songs in this data set are rock songs (to 1 d.p.)?

count1=data['RockOrNot'].value_counts()[1]
round((count1/len(data))*100,1)

#Ans : 48.8% of the songs are rock in the given data sample.


# In[4]:


# Q2 To perform a classification algorithm, you need to define a classification 
# variable and separate it from the other variables. We will use 'RockOrNot' as 
# our classification variable. Write a piece of code to separate the data into a 
# DataFrames X and a Series y, where X contains a standardised version of 
# everything except for the classification variable ('RockOrNot'), and y contains 
# only the classification variable. To standardise the variables in X, you need
# to subtract the mean and divide by the standard deviation

def classification(df,ind_col):
    y=pd.Series(df[ind_col])
    df1=df.drop([ind_col],axis=1)
    X=(df1-df1.mean())/df1.std()
    return(X,y)


# In[5]:


X,y=classification(data,'RockOrNot')


# In[6]:


# Q3 Which variable in X has the largest correlation with y?

X.corrwith(y).idxmax()


#Ans:PAR_SFM_M is the variable which is highly correlated with y variable.


# In[7]:


# Q4 When performing a classification problem, you fit the model to a portion of 
# your data, and use the remaining data to determine how good the model fit was.
# Write a piece of code to divide X and y into training and test sets, use 75%
# of the data for training and keep 25% for testing. The data should be randomly
# selected, hence, you cannot simply take the first, say, 3000 rows. If you select 
# rows 1,4,7,8,13,... of X for your training set, you must also select rows 
# 1,4,7,8,13,... of y for training set. Additionally, the data in the training
# set cannot appear in the test set, and vice versa, so that when recombined,
# all data is accounted for. Use the seed 123 when generating random numbers
# Note: The data may not spilt equally into 75% and 25% portions. In this 
# situation you should round to the nearest integer.

def splitting(X,y,per,s):
    trainx=X.sample(frac=per,random_state=s) 
    testx=X.drop(trainx.index)
    trainy=y.sample(frac=per,random_state=s)
    testy=y.drop(trainy.index)
    return trainx,testx,trainy,testy


# In[8]:


train_x,test_x,train_y,test_y=splitting(X,y,0.75,123)


# In[9]:


# Q5 What is the percentage of rock songs in the training dataset and in the 
# test dataset? Are they the same as the value found in Q1?

#Ans : No, the values are not same as found in Q1,when applied to train set
#it differs by 1 unit more/less.49.4% of
#the songs are rock in the train dataset and 47.09% of the songs are rock in test dataset.

((train_y.value_counts()[1])/len(train_y))*100


# In[10]:


((test_y.value_counts()[1])/len(test_y))*100


# In[15]:


# Q6 Now we're going to write a function to run kNN on the data sets. kNN works 
# by the following algorithm:
# 1) Choose a value of k (usually odd)
# 2) For each observation, find its k closest neighbours
# 3) Take the majority vote (mean) of these neighbours
# 4) Classify observation based on majority vote

def kNN(X,y,k):
    n=len(X)
    y1=y.tolist()
    y_star=[]
    dist=squareform(pdist(X))
    for i in range(n):
        dist[i][i]=float('inf')
    for i in range(n):
        d=[]
        d=dist[i]
        ind=[]
        ind=sorted(range(len(d)),key=lambda s:d[s])
        y_nearest=[]
        for j in range(k):
            y_nearest.append(y1[ind[j]])
        c1=y_nearest.count(1)
        c0=y_nearest.count(0)
        if(c1>c0):
            y_star.append(1)
        else:
            y_star.append(0)
    ystar=pd.Series(y_star)
    return ystar


# In[16]:


# Q7 The misclassification rate is the percentage of times the output of a 
# classifier doesn't match the classification value. Calculate the 
# misclassification rate of the kNN classifier for X_train and y_train, with k=3.

def misClassRate(X,y,k):
    y_star1=pd.Series()
    y_star1=kNN(X,y,k)
    n=len(y)
    y1=y.tolist()
    counter=0
    for i in range(n):
        if(y_star1[i] != y1[i]):
            counter=counter+1
    rate=(counter/n)*100
    return rate


# In[17]:


rates=misClassRate(train_x,train_y,3)
rates

#Ans:The misclassification rate of train set when the kvalue=3 for train data is 4.70


# In[18]:


# Q8 The best choice for k depends on the data. Write a function kNN_select that 
# will run a kNN classification for a range of k values, and compute the 
# misclassification rate for each.

def kNN_select(X,y,k_vals):
    rates=[]
    kvalues=[]
    for k in k_vals:
        rate=0
        rate=misClassRate(X,y,k)
        rates.append(rate)
        kvalues.append(k)
    mis_class_rates=pd.Series(rates,index=kvalues)
    return mis_class_rates


# In[19]:


# Q9 Run the function kNN_select on the training data for k = [1, 3, 5, 7, 9] 
# and find the value of k with the best misclassification rate. Use the best 
# value of k to report the mis-classification rate for the test data. What is 
# the misclassification percentage with this k on the test set?


kvalue=kNN_select(train_x,train_y,[1, 3, 5, 7, 9]).idxmin()

#Ans:kvalue is 1 which produces least misclassification rate.

misClassRate(test_x,test_y,kvalue)

#Ans:The misclassification rate with kvalue=1 is 5.0


# In[20]:


# Q10 Write a function to generalise the k nearest neighbours classification 
# algorithm. 

def kNN_classification(df,class_column,seed,percent_train,k_vals):
    X,y=classification(df,class_column)
    trainx,testx,trainy,testy=splitting(X,y,percent_train,seed)
    misrates=kNN_select(trainx,trainy,k_vals)
    k=misrates.idxmin()
    mis_class_test=misClassRate(testx,testy,k)
    return mis_class_test


# In[21]:


# Test your function with the TunedIT data set, with class_column = 'RockOrNot',
# seed = the value from Q4, percent_train = 0.75, and k_vals = set of k values
# from Q8, and confirm that it gives the same answer as Q9.

kNN_classification(data,'RockOrNot',123,0.75,[1, 3, 5, 7, 9])

#Ans: The misclassification rate is 5.0 which is same answer as Q9.


# In[22]:


data1=pd.read_csv('house_votes.csv')


# In[23]:


# Now test your function with another dataset, to ensure that your code 
# generalises.

kNN_classification(data1,'Party',123,0.75,[1, 3, 5, 7, 9])

#Ans: The misclassification rate is ~8% and precisely 8.25.


# In[ ]:




