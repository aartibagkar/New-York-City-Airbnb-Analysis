#!/usr/bin/env python
# coding: utf-8

# In[159]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[160]:


airbnb=pd.read_csv('C:/Project/Airbnb Analysis/AB_NYC_2019.csv')


# Understand the data

# In[161]:


airbnb.head()


# In[162]:


airbnb.info()


# Prepocess the data

# In[163]:


airbnb.drop_duplicates(inplace=True)

#Check for null values 

airbnb.isnull().sum()

airbnb.dropna(how='any',inplace=True)


# In[164]:


airbnb.fillna({'reviews_per_month':0}, inplace=True)
airbnb.reviews_per_month.isnull().sum()


# Understand Correlation between different variable

# In[165]:


cor = airbnb.corr(method="pearson") #Calculate the correlation of the above variables
plt.figure(figsize=(15,8))
sns.heatmap(cor, square = True,annot=True) 


# Analyze most stayed location

# In[166]:


airbnb['neighbourhood_group'].unique()


# In[167]:


sns.countplot(airbnb['neighbourhood_group'])


# In[168]:


print("Brooklyn and Manhattan are having high Airbnb booking count")


# In[169]:


sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.neighbourhood_group)


# understand the relationship between no of reviews per neighbourhood
# 

# In[170]:


# fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
op = airbnb.groupby(['neighbourhood_group']).count()['number_of_reviews']



# Analyzing the dependency between variables

# In[171]:


plt.figure(figsize=(15,12))
sns.scatterplot(x='room_type', y='price', data=airbnb)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price",size=15, weight='bold')


# Understanding Multicollinearity
# Understand the relationship between variables in multiple regression. 
# If there is multicollinearity occurs, these highly related input variables should be eliminated from the model.

# In[172]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

airbnb_new = airbnb
airbnb_new.drop(['name','id','host_name','last_review','host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)

airbnb_new.loc[:,"neighbourhood_group"] =le.fit_transform(airbnb_new["neighbourhood_group"])
airbnb_new.loc[:,"room_type"] =le.fit_transform(airbnb_new["room_type"])



airbnb_new.head()


# In[150]:


#Get Correlation between different variables
corr = airbnb_new.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)


# In[152]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


# In[218]:


#Defining the independent variables and dependent variables
# airbnb_new.loc[:,"neighbourhood_group"]
# x = airbnb_new.loc[:,["neighbourhood_group",'room_type','minimum_nights','calculated_host_listings_count','availability_365']]
x = airbnb_new.loc[:,['neighbourhood_group','room_type','minimum_nights','calculated_host_listings_count','availability_365']]
y = airbnb_new.loc[:,'price']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=353)


# In[219]:


#Prepare a Linear Regression Model
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)



# In[220]:


#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[ ]:




