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


# # Data Preprocessing

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


# In[17]:


df.n_hot_rooms[(df.n_hot_rooms)>3*uppervalue]=3*uppervalue


# In[18]:


df[(df.n_hot_rooms)>uppervalue]  


# In[19]:


np.percentile(df.rainfall,[1])


# In[20]:


lowervalue = np.percentile(df.rainfall,[1])[0]


# In[21]:


df[(df.rainfall)<lowervalue]


# In[22]:


df.rainfall[(df.rainfall)<0.3*lowervalue]=0.3*lowervalue


# In[23]:


df[(df.rainfall)<lowervalue]


# In[24]:


sns.jointplot(x="crime_rate",y="price",data=df)


# In[25]:


df.describe()


# In[26]:


df.n_hos_beds = df.n_hos_beds.fillna(df.n_hos_beds.mean())


# In[27]:


df.info


# In[28]:


df.info()


# -> Variable Transformation 

# In[29]:


sns.jointplot(x='crime_rate',y='price',data=df)


# This scatter plot shows a curve which looks like logarithmic relation between price and crime rate. We need to transform this curve to linear curve in order to get a linear relation. Also most of the values are near zero and log of zero is not defined.So to remove this we will add a value that is 1.

# In[30]:


df.crime_rate=np.log(1 + df.crime_rate)


# In[31]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[32]:


df['avg_dist']=(df.dist1 + df.dist2 + df.dist3 + df.dist4)/4


# In[33]:


df.describe()


# In[34]:


del df['dist1']


# In[35]:


del df['dist2']


# In[36]:


del df['dist3']


# In[37]:


del df['dist4']


# In[38]:


df.head()


# In[39]:


del df['bus_ter']


# In[40]:


df.head()


# In[41]:


df = pd.get_dummies(df)


# In[42]:


df.head()


# In[43]:


del df['airport_NO']


# In[44]:


del df['waterbody_None']


# In[45]:


df.head()


# ### Understanding Correlation 

# In[47]:


df.corr()


# In[48]:


del df['parks']


# In[49]:


df.head()


# In[ ]:




