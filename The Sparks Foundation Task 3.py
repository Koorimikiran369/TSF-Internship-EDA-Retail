#!/usr/bin/env python
# coding: utf-8

# #  Task 3: Exploratory Data Analysis-Retail
# ## - By Mayank Sharma

# #### Objective: To analyse the SampleSuperstore dataset which contains data about a superstore and the sales done along with some of the factors and their corresponding profits.

# ## Importing Libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Exploring the dataset

# In[4]:


data=pd.read_csv(r'C:\Users\Mayank Sharma\Desktop\SampleSuperstore.csv',encoding='latin',low_memory='False')


# In[5]:


data.head() #looking through the table


# In[6]:


data.shape #checking the shape of table


# In[7]:


data.columns #list of columns


# In[8]:


data.describe() #a brief summary of numerical variables


# In[10]:


data.info() #check the datatype of the columns


# ## Data Preprocessing

# In[11]:


data.isnull().sum() #looking for null values


# In[24]:


data.nunique() #looking for unique values in each columns


# In[14]:


data.duplicated().sum() #looking for duplicate items in table


# In[15]:


data.loc[data.duplicated(),:] #displaying duplicate items in table


# In[16]:


data.drop_duplicates(inplace=True) #removing duplicate items


# In[17]:


data.shape #we see that numbers of row are reduced now


# ### Splitting the table on the basis of Profit and Loss

# In[18]:


data_profit=data.loc[data.Profit>=0,:] #profit table
data_loss=data.loc[data.Profit<0,:] #loss table


# In[19]:


data_profit.shape 


# In[20]:


data_loss.shape


# In[21]:


data_loss['Profit']=data_loss['Profit'].abs() #changing negative sign to positive loss value


# In[22]:


data_loss.head()
data_loss.rename(columns={'Profit':'Loss'},inplace=True) #renaming profit to loss in loss table


# ## Data Visualization and EDA

# In[26]:


fig, axes = plt.subplots(2, 2, figsize=(18, 9)) #plotting the count plots 
fig.suptitle('Contributers with the help of count plots')
sns.countplot(ax=axes[0][0], data=data_profit, x='Ship Mode')
sns.countplot(ax=axes[0][1], data=data_profit, x='Region')
sns.countplot(ax=axes[1][0], data=data_loss,x='Category')
sns.countplot(ax=axes[1][1], data=data_profit, x='Sub-Category')
plt.xticks(rotation=45) #to rotate the x labels by 45 degree
plt.show()


# The above subplots show the count of different categorical variables which we will furthur use for EDA.

# In[278]:


data.corr()


# In[357]:


plt.figure(figsize=(9,5)) 
sns.heatmap(data.corr(),annot=True) #heatmap
plt.show()


# We see that sales and profit are postively correlated but the relationship is not that strong. Also discount and profit have week negative correlation.

# In[27]:


plt.figure(figsize=(9,5))
sns.scatterplot(x='Sales',y='Profit',data=data) #scatterplot
plt.show()


# It is observed from the plot that the profit increases with sales after a certain range. At very higher sales the profit is more but we usually consider them as outliers in the above graph.

# In[28]:


sns.set_style('darkgrid')
sns.set(context='notebook',
    style='darkgrid',
    palette='gnuplot',
    font='sans-serif',
    font_scale=1,
    rc=None) #plot settings


# In[32]:


plt.figure(figsize=(9,5))
sns.barplot(x='Region',y='Profit',data=data,estimator=sum,ci=None)
plt.show()


# The western and eastern are the most profitable regions.

# In[31]:


fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Category with maximum profit and loss')
sns.barplot(ax=axes[0], data=data_profit, x='Category', y='Profit',estimator=sum,ci=None)
sns.barplot(ax=axes[1], data=data_loss,x='Category', y='Loss',estimator=sum,ci=None)
plt.show()


# We observe that technology category accounts for more profit and furniture for the losses.

# In[34]:


fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Subcategory with maximum profits and losses')
sns.barplot(ax=axes[0], data=data_profit.sort_values('Profit',ascending=False), x='Profit', y='Sub-Category',ci=None)
sns.barplot(ax=axes[1], data=data_loss.sort_values('Loss',ascending=False), x='Loss', y='Sub-Category',ci=None)
plt.show()


# The mean profit for copiers is maximum and is minumum for machines.

# In[62]:


samp1 = data.groupby("Segment")["Profit"].sum() #preparing data for pie chart
samp2 = data.groupby("Ship Mode")["Profit"].sum() #preparing data for pie chart


# In[56]:


plt.figure(figsize=(16,9))
samp1.plot.pie(autopct="%.1f%%")
plt.show()


# The consumer segment provides the maximum part of the profit

# In[64]:


plt.figure(figsize=(16,9))
samp2.plot.pie(autopct="%.1f%%")
plt.show()


# As we have seen earlier that Standard Class is most active in previous count plots, here it becomes clear that it also accounts for most of the profits too.

# In[65]:


plt.figure(figsize=(16,9))
samp_data=data.groupby(["State"])["Profit"].sum()
samp_data.sort_values(ascending=False,inplace=True)
samp_data.plot.bar()
plt.show()


# The states like California,New York show a good amount of profit while states like Texas and Ohio show some poor results of the same.

# ### We have performed the exploratory data analysis of the dataset. This can be used to for the feature engineering part to get new features if we want to train the model to predict the profits on an unknown test data.
