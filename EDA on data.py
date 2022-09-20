#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv("E:\BYOP project/energy_demand - Copy.csv")


# In[3]:


# Understanding the data

data.head()


# In[5]:


# Details : Five years of energy price and demand data (source):

"""You have access to over five years of energy price and demand data :

date - from January 1, 2015, to October 6, 2020.
demand - daily electricity demand in MWh.
price - recommended retail price in AUD/MWh.
demand_pos_price - total daily demand at a positive price in MWh.
price_positive - average positive price, weighted by the corresponding intraday demand in AUD/MWh.
demand_neg_price - total daily demand at a negative price in MWh.
price_negative - average negative price, weighted by the corresponding intraday demand in AUD/MWh.
frac_neg_price - the fraction of the day when the demand traded at a negative price.
min_temperature - minimum temperature during the day in Celsius.
max_temperature - maximum temperature during the day in Celsius.
solar_exposure - total daily sunlight energy in MJ/m^2.
rainfall - daily rainfall in mm.
school_day - "Y" if that day was a school day, "N" otherwise.
holiday - "Y" if the day was a state or national holiday, "N" otherwise."""


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


# 2106 rows and 14 columns


# In[11]:


data.describe  #all the count values/ Integers value showing up here 


# In[13]:


data.columns #all the columns


# In[15]:


data.nunique()


# In[16]:


#cleaning the data


# In[17]:


data.isnull().sum()


# In[22]:


data = data.dropna()


# In[23]:


data.isnull().sum()


# In[ ]:




