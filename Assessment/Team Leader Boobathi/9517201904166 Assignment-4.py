#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[2]:


data = pd.read_csv('abalone.csv')


# In[3]:


# Univariate Analysis
plt.hist(data['Sex']);
plt.xlabel('Sex');


# In[24]:


plt.hist(data['Rings']);
plt.xlabel('Rings');


# In[5]:


sns.boxplot(x=data['Length'])
plt.xlabel('Length');


# In[6]:


plt.hist(data['Diameter']);
plt.xlabel('Diameter');


# In[8]:



plt.hist(data['Height']);
plt.xlabel('Height');


# In[9]:


sns.boxplot(x=data['Whole weight'])
plt.xlabel('Whole weight');


# In[10]:


sns.boxplot(x=data['Shucked weight'])
plt.xlabel('Shucked weight');


# In[11]:


plt.hist(data['Viscera weight']);
plt.xlabel('Viscera weight');


# In[12]:


# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.lineplot(x=data["Height"], y=data["Whole weight"]);
plt.xlabel('Height');
plt.ylabel('Whole weight');


# In[13]:


plt.figure(figsize=(10, 6))
sns.lineplot(x=data["Length"], y=data["Height"]);
plt.xlabel('Length');
plt.ylabel('Height');


# In[14]:


plt.figure(figsize=(10, 6))
sns.lineplot(x=data["Diameter"], y=data["Height"]);
plt.xlabel('Diameter');
plt.ylabel('Height');


# In[15]:


plt.figure(figsize=(10, 6))
sns.lineplot(x=data["Length"], y=data["Diameter"]);
plt.xlabel('Length');
plt.ylabel('Diameter');


# In[16]:


plt.figure(figsize=(10, 6))
plt.scatter(x=data["Shucked weight"], y=data["Whole weight"]);
plt.xlabel('Shucked weight');
plt.ylabel('Whole weight');


# In[17]:


plt.figure(figsize=(10, 6))
plt.scatter(x=data["Viscera weight"], y=data["Whole weight"]);
plt.xlabel('Viscera weight');
plt.ylabel('Whole weight');


# In[18]:


# Multi-variate Analysis
plt.figure(figsize=(10, 6));
sns.heatmap(data.corr(), annot=True);


# In[25]:


# Handling Missing Values

data.isna().sum()


# In[26]:


# Descriptive Statistics

data.describe()


# In[28]:


# Outlier Handling

numeric_cols = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']


# In[29]:


def boxplots(cols):
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))

    t=0
    for i in range(4):
        for j in range(2):
            sns.boxplot(ax=axes[i][j], data=data, x=cols[t])
            t+=1

    plt.show()


# In[30]:


def Flooring_outlier(col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    whisker_width = 1.5
    lower_whisker = Q1 -(whisker_width*IQR)
    upper_whisker = Q3 + (whisker_width*IQR)
    data[col]=np.where(data[col]>upper_whisker,upper_whisker,np.where(data[col]<lower_whisker,lower_whisker,data[col]))

print('Before Outliers Handling')
print('='*100)
boxplots(numeric_cols)
for col in numeric_cols:
    Flooring_outlier(col)
print('\n\n\nAfter Outliers Handling')
print('='*100)
boxplots(numeric_cols)


# In[31]:


# Encode Categorical Columns

data = pd.get_dummies(data, columns = ['Sex'])
data


# In[32]:


# Split Data into Dependent & Independent Columns
Y = data[['Rings']]
X = data.drop(['Rings'], axis=1)


# In[33]:


# Scale the independent Variables

scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# In[34]:


# Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[35]:


# Model Training & Testing

model = LinearRegression()
model.fit(X_train, Y_train)
model.score(X_train, Y_train), model.score(X_test, Y_test)

model = DecisionTreeRegressor(max_depth=15, max_leaf_nodes=40)
model.fit(X_train, Y_train)
model.score(X_train, Y_train), model.score(X_test, Y_test)

