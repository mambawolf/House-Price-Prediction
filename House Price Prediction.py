#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# ### We will use Linear regression model to predict the house prices based on certain attributes.

# In[2]:


pwd //to know about the working directory


# ## Importing the dataset 

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[4]:


df = pd.read_csv("House_Price.csv", header=0)


# In[5]:


df.head()


# In[7]:


df.shape


# ### EDD : Extended data dictionary 

# In[8]:


df.describe()


# In[10]:


sns.jointplot(x="n_hot_rooms", y="price", data=df)


# In[11]:


sns.jointplot(x="rainfall", y="price", data=df)


# In[12]:


sns.countplot(x="airport",data=df)


# In[13]:


sns.countplot(x="waterbody",data=df)


# In[14]:


sns.countplot(x="bus_ter",data=df)


# In[ ]:




