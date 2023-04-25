#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


df = pd.read_csv('insurance.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


sns.set()
plt.figure(figsize =(6,6))
sns.distplot(df['age'])
plt.title("Age Distribution")
plt.show()


# In[10]:


plt.figure(figsize =(6,6))
sns.countplot(x = "sex", data =df)
plt.title("Sex Distribution")
plt.show()


# In[11]:


df['sex'].value_counts()


# In[12]:


sns.distplot(df['bmi'])
plt.show()


# In[13]:


df['region'].value_counts()


# In[14]:


df.replace({'sex':{'male':0,'female':1}},inplace= True)


# In[16]:


df.replace({'smoker':{'yes':0,'no':1}},inplace = True)


# In[18]:


df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace = True)


# In[20]:


x = df.drop(columns= "charges",axis=1)
y=df["charges"]


# In[21]:


x


# In[22]:


y


# In[23]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)


# In[24]:


x_train.shape


# In[25]:


x_test.shape


# In[26]:


reg = LinearRegression()


# In[27]:


reg.fit(x_train,y_train)


# In[28]:


training_data_prediction= reg.predict(x_train)


# In[29]:


r2_train= metrics.r2_score(y_train,training_data_prediction)


# In[30]:


r2_train


# In[31]:


test_data_prediction = reg.predict(x_test)


# In[32]:


metrics.r2_score(y_test,test_data_prediction)


# In[33]:


sample_input_data= (30,1,22.7,0,1,0)


# In[34]:


input_data_as_numpy_array = np.asarray(sample_input_data)


# In[35]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[36]:


prediction = reg.predict(input_data_reshaped)


# In[37]:


print("The insurance cost is ", prediction)


# In[ ]:




