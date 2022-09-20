#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from itertools import product
from typing import Union
from tqdm.notebook import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error as MAPE
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

sns.set_style('white')


# In[3]:


df = pd.read_csv('E:\IPBA\BYOP project/energy_demand.csv', parse_dates=['date'])


# In[ ]:


#Exploratory Data Analysis 

1- Electricity demand is slowly shrinking, even before the pandemics. Our data unable to explain this behavior, but some web articles (mentioned below ), 
    point to renewable generation.
    
    https://www.energycouncil.com.au/analysis/increases-in-negative-prices-is-it-a-positive/
        
2- Prices are higher at both lower and higher temperatures. It usually hits a peak at the end of the summer, but it's more stable during winter.

3- Electricity prices are higher in normal school days, which usually means commercial days.

4- Only 191 days in the whole period traded electricity at a negative price,however 
   they are not evenly distributed throughout the period and have become more frequent in recent years. 
5- The most days that had negative electricity prices only traded them for less than 25% of the day, or six hours.


# In[4]:


# evolution of energy demand on a yearly basis
from statsmodels.tsa.seasonal import STL
decomposition = STL(df['demand'], period=365).fit()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                               figsize=(12,9), sharex=True)

ax1.plot(decomposition.trend)
ax1.set_ylabel('Trend of Demand (MWh)')
ax1.text(1700, 120000, s=
        """This line shows that demand has been
falling since 2015, except for a small period
during 2017.""", fontsize=14)

ax2.plot(decomposition.seasonal)
ax2.set_ylabel('Seasonal')

plt.xticks(np.arange(0, 2100, 360), np.arange(2015, 2021, 1))
fig.autofmt_xdate()
sns.despine()
plt.show()


# # Colder & hotter than average days increasing electricity consumption :

# In[8]:


# minimum and maximum temperature x price
data = df[['date', 'RRP', 'max_temperature', 'min_temperature']]
data['avg_temperature'] = (data['max_temperature'] + data['min_temperature'])/2

# RRP is recommended retail price in AUD$ / MWh


# In[9]:


fig, ax = plt.subplots(figsize=(10,7))

sns.regplot(data=data, x='avg_temperature', y='RRP', order=2, ax=ax,
            line_kws={'color': 'red'})

ax.set_xlabel('Temperature (º C)')
ax.set_ylabel('RRP (AUD/MWh)')

ax.text(33, 600,
                    """We can see that lower temperatures are 
slightly related to higher prices, but
this relation is stronger with higher 
temperatures.

Avg. Temperature was calculated by
summing min and max temperatures
divided by 2.""", fontsize=14)

plt.title('RRP x Average Temperature', fontsize=14)
plt.ylim(-20, 1000)
sns.despine()
plt.show()


# # Prices behave differently on different day types (holidays, schooldays, weekends)

# In[11]:


#electricity consumption on different day types
# RRP is recommended retail price in AUD$ / MWh
data = df[['RRP', 'school_day', 'holiday']]

g = sns.catplot(data=data, x='school_day', y='RRP', kind='box',
                showfliers=False, col='holiday', col_order=['N', 'Y'],
               aspect=12/15)

g.set_axis_labels('School Day', 'RRP (AUD/MWh)')
g.set_titles('Is holiday? {col_name}')

g.axes.flat[0].text(3.8, 135,
"""It seems prices (RRP) are higher on
school days, whether it's a national holiday
or not. Despite not showing outliers, notice
how the only category on which prices (RRP) dip 
below 0 is 'days which are neither school
days nor holidays'.""", fontsize=14)

plt.suptitle("Price (RRP) vs Type of Day", fontsize=15, y=1.08)
plt.show()


# In[18]:


#electricity consumption on different day types
data = df[['RRP', 'school_day', 'date']]
data['weekday'] = ['Y' if x <= 4 else 'N' for x in df['date'].dt.weekday]

g = sns.catplot(data=data, x='weekday', y='RRP', kind='box',
                showfliers=False, aspect=12/15)

g.set_axis_labels('Week Day?', 'RRP (AUD/MWh)')

g.axes.flat[0].text(1.8, 135,
"""This plot shows the distribution
of prices in weekends/weekdays. It shows
that prices are usually lower on weekends,
while it doesn't tell us much about negative
prices.""", fontsize=14)

plt.suptitle("Prices on Weekdays/Weekends", fontsize=15, y=1.08)
plt.show()


# # Rainfall doesn't seem to strongly influence our prices

# In[24]:


# minimum and maximum temperature x price
# RRP is recommended retail price in AUD$ / MWh

data = df[['RRP', 'rainfall']]

g = sns.lmplot(data=data[data['rainfall'] > 0], x='rainfall', y='RRP',
                height=6, aspect=10/6, order=1)

g.set_axis_labels('Rainfall (mm)', 'RRP (AUD/MWh)')
g.set(ylim=(-20, 300))

plt.title('Rainfall vs RRP', fontsize=14)
plt.show()


# # However, solar exposure seems to be related:

# In[27]:


# minimum and maximum temperature x price

data = df[['RRP', 'solar_exposure']]

g = sns.lmplot(data=data, x='solar_exposure', y='RRP',
                height=6, aspect=10/6, order=2, line_kws={'color': 'red'})

g.set_axis_labels('Solar Exposure (MJ/m²)', 'RRP (AUD/MWh)')
g.set(ylim=(-20, 300))

plt.title('Solar Exposure vs Price (RRP)', fontsize=14)
plt.show()


# In[ ]:





# # There are only 191 days in which electricity traded for negative prices, and...

# In[30]:


filt = df['frac_at_neg_RRP'] > 0

fig, ax = plt.subplots(figsize=(8,6))

x = ['Non-negative', 'Negative']
y = [df[~filt].shape[0], df[filt].shape[0]]
ax.bar(x=x,
      height=y)

ax.set_xlabel("Electricity Prices")
ax.set_ylabel("Count")

for i, value in enumerate(y):
    plt.text(x=i, y=value+1, s=str(value), ha='center')

plt.suptitle("Days in which electricity price was...", y=0.93, fontsize=15)
plt.show()


# # Most days only traded negatively for less than 6 hours. However...
# 
# We see that most days with negative prices are only so for less than 6 hours (roughly 93.71%). Nevertheless, it's a substantial fraction of the day that can lead to bad results if not anticipated. Now that we know for how long during the day electricity trades for negatively, it's time to understand why it does so. Is there a pattern for these occurrences? One thing to notice is how there was a surge in these points from mid-2019 onwards, perhaps indicating a public move towards renewable energy (solar panels at home, more so as pandemic restrictions were enforced and led to homeworking).

# In[33]:


# negative prices throughout data

def map_fraction_of_day(frac_neg):
    frac_dict = {1/24: '<= 1h',
                1/4: '>1h and <= 6h',
                1/2: '>6h and <= 12h',
                1: '>12h'}
    
    for key in frac_dict.keys():
        if frac_neg <= key:
            return frac_dict[key]

data = df[filt]
data['neg_duration'] = data['frac_at_neg_RRP'].apply(lambda x: map_fraction_of_day(x))

col_pal = sns.color_palette("colorblind", 4)
temp = col_pal[1]
col_pal[1] = col_pal[2]
col_pal[2] = temp

g = sns.relplot(data=data, x='date', y='RRP_negative', hue='neg_duration', kind='scatter',
            height=6, aspect=11/6, palette=col_pal,
           s=35, alpha=0.7)

g.set_axis_labels('Date', 'Neg. Price (AUD/MWh)')
g.legend.set_bbox_to_anchor([1.1, 0.5])
g._legend.set_title("Dur. of Negative Prices")

plt.ylim(1, -100)
plt.setp(g._legend.get_texts(), fontsize=12)
plt.setp(g._legend.get_title(), fontsize=14)

fig.autofmt_xdate()
plt.show()


# In[ ]:




