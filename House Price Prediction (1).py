#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# ### We will use Linear regression model to predict the house prices based on certain attributes.

# In[1]:


pwd //to know about the working directory


# ## Importing the dataset 

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


df = pd.read_csv("House_Price.csv", header=0)


# In[4]:


df.head()


# In[5]:


df.shape


# ### EDD : Extended data dictionary 

# In[6]:


df.describe()


# In[7]:


sns.jointplot(x="n_hot_rooms", y="price", data=df)


# In[8]:


sns.jointplot(x="rainfall", y="price", data=df)


# In[9]:


sns.countplot(x="airport",data=df)


# In[10]:


sns.countplot(x="waterbody",data=df)


# In[11]:


sns.countplot(x="bus_ter",data=df)


# Observations
# 
# 1. There are missing values in n-hos-bed as total should be 506 but there are only 498 values
# 2. There is skewness or outlier in crime rate
# 3. There are outliers in both n_hot_rooms and rainfall
# 4. bus_ter has only 'Yes' values

# ### Outlier Treatment 

# In[12]:


df.info()


# In[13]:


np.percentile(df.n_hot_rooms,[99])


# In[14]:


np.percentile(df.n_hot_rooms,[99])[0]


# In[15]:


uppervalue=np.percentile(df.n_hot_rooms,[99])[0]


# In[16]:


df[(df.n_hot_rooms)>uppervalue]


# In[18]:


df.n_hot_rooms[(df.n_hot_rooms)>3*uppervalue]=3*uppervalue


# In[19]:


df[(df.n_hot_rooms)>uppervalue]  //Now the values 101 and 81 are made to 46 and 46 


# In[20]:


np.percentile(df.rainfall,[1])


# In[21]:


lowervalue = np.percentile(df.rainfall,[1])[0]


# In[22]:


df[(df.rainfall)<lowervalue]


# In[23]:


df.rainfall[(df.rainfall)<0.3*lowervalue]=0.3*lowervalue


# In[24]:


df[(df.rainfall)<lowervalue]


# In[25]:


sns.jointplot(x="crime_rate",y="price",data=df)


# In[26]:


df.describe()


# In[ ]:




