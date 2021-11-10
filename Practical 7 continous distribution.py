#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.display import Math, Latex
from IPython.core.display import Image
import numpy as np


# In[13]:


import seaborn as sns
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(5,5)})


# # Uniform Distribution
# 

# In[9]:


from scipy.stats import uniform


# In[10]:


n = 1000
start = 10
width = 20
data_uniform = uniform.rvs(size=n, loc = start, scale=width)


# In[17]:


ax = sns.distplot(data_uniform,
                bins=100,
                kde=True,
                color='skyblue',
                hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Uniform Distribution', ylabel='Frequency')


# # Normal Distribution

# In[16]:


from scipy.stats import norm
data_normal = norm.rvs(size=1000,loc=0,scale=1)


# In[18]:


ax = sns.distplot(data_normal,
                 bins=100,
                 kde=True,
                 color='skyblue',
                 hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')


# # Exponatial Distribution

# In[22]:


from scipy.stats import expon
data_expon = expon.rvs(size=1000,loc=0,scale=1)


# In[23]:


ax = sns.distplot(data_expon,
                 bins=100,
                 kde=True,
                 color='skyblue',
                 hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Exponatial Distribution', ylabel='Frequency')


# # Chi Square Distribution

# In[24]:


from numpy import random

x = random.chisquare(df=2, size=(2, 3))

print(x)


# In[25]:


from numpy import random
import matplotlib.pyplot as plt 
import seaborn as sns
sns.distplot(random.chisquare(df=1, size=1000), hist=False)
plt.show()
    


# # Weibull Distribution

# In[26]:


a = 5. #shape

s = np.random.weibull(a,100)


# In[28]:


import matplotlib.pyplot as plt

x = np.arange(1,100.)/50.

def weib(x,n,a):
    
    return (a / n)*(x / n)**(a - 1)*np.exp(-(x / n)**a)


# In[29]:


count, bins, ignored = plt.hist(np.random.weibull(5,1000))

x = np.arange(1,100.)/50.

scale = count.max()/weib(x, 1., 5.).max()

plt.plot(x, weib(x,1.,5.)*scale)

plt.show()


# In[ ]:




