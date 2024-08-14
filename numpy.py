#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
a=np.array([1,2,3])
a


# In[2]:


b=np.array([1,2,3])
b


# In[3]:


a=np.array([1,2,3])
a
a[2:4]


# In[5]:


a=np.array([1,2,3])
a
a[2;4]


# In[7]:


import numpy as np
a=np.array([[1,2,3,],[4,5,6]])
print(a)


# In[8]:


a=np.array([[1,2,3],[4,5,6],[7,8,9]],float)
print(a)


# In[9]:


a[2][0]


# In[10]:


a[0:2][1]


# In[11]:


a[0:2]


# In[12]:


a[2][1]


# In[16]:


import numpy as np
b=np.full((2,4),15)
b


# In[3]:


import numpy as np
a=np.arange(20)
a


# In[4]:


a.size


# In[12]:


a.itemsize


# In[11]:


a.itemsize
a.dtype


# In[17]:


b.ndim


# In[18]:


a.nbytes


# In[19]:


c=np.array([1])
c.nbytes


# In[21]:


c=np.array([1,2])
c.nbytes


# In[22]:


a=np.array([1,2,3,4,5])
a


# In[23]:


a.dtype


# In[25]:


a.astype('int8')


# In[26]:


a=np.array([2,4,6,8,10])
a


# In[27]:


np.append(a,12)


# In[51]:


a=np.array([2,4,6,8,10])
a


# In[52]:


a=np.insert(a,2,5)


# In[53]:


import numpy as np
a=np.array([[1,2,30],[4,5,6]])
a


# In[54]:


np.insert(a,2,5,axis=1)
a


# In[55]:


np.insert(a,2,5,axis=0)


# In[56]:


np.delete(a,1,axis=1)


# In[57]:


np.delete(a,1,axis=0)


# In[58]:


np.delete(a,0,axis=1)


# In[61]:


a=np.random.random(5)
a


# In[63]:


a=np.random.randint(1,10,5)
a


# In[65]:


a=np.random.randint(1,10)
a


# In[68]:


a=np.random.randint(2,10)
a


# In[70]:


a=np.random.randint(2,10,[2,2])
a


# In[4]:


import numpy as np
a=np.random.randint(1,10,(10,10))
                    
a


# In[11]:


a=np.full((3,3),True)
a


# In[12]:


a=np.random.randint(10,100,25)
a


# In[13]:


a=np.array([1,2,3])
b=np.copy(a)
b


# In[14]:


c=a.copy()
c


# In[15]:


a=np.array([1,2,3,5,7,6,4])
np.sort(a)


# In[16]:


b=np.array([[5,8,1],[4,6,2]])
print(b)


# In[17]:


np.sort(b,axis=0)


# In[18]:


np.sort(b,axis=1)


# In[20]:


a=np.array([3,4,5,6])
a


# In[21]:


a+2


# In[22]:


a-2


# In[23]:


a*2


# In[24]:


a**2


# In[25]:


a/2


# In[33]:


a=np.array([[1,2,3],[5,2,1]])
print(a)
b=np.array([[2,3,1],[4,3,2]])
print(b)


# In[34]:


np.add(a,b)


# In[35]:


np.substract(a,b)


# In[36]:


np.divide(a,b)


# In[37]:


np.multiply(a,b)


# In[38]:


a=np.array([[1,2,3],[4,5,6]])
print(a)
b=np.array([[4,5,6],[1,4,5]])
print(b)


# In[39]:


a+b


# In[40]:


a-b


# In[41]:


a/b


# In[42]:


a*b


# In[43]:


a=np.array([[1,2,3],[4,5,6]])
print(a)
b=np.array([[4,5,6],[1,4,5]])
print(b)


# In[44]:


np.vstack([a,b])


# In[45]:


np.hstack([a,b])


# In[46]:


np.concatenate([a,b],axis=0)


# In[48]:


np.concatenate([a,b],axis=1)


# In[50]:


a=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
a


# In[51]:


a[0,1]


# In[52]:


a[2,2]


# In[53]:


a[2,2]


# In[54]:


a[1,4]


# In[55]:


a[2,0]


# In[56]:


a[0:]


# In[57]:


a[0:2]


# In[58]:


a[0:3]


# In[59]:


a[0:2,1:3]


# In[60]:


a[1:3,1:4]


# In[61]:


a[0:2,1:2]


# In[6]:


import numpy as np
a=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30]])
a


# In[67]:


a[3:4]


# In[68]:


a[0:3]


# In[69]:


a[0:3,4]


# In[70]:


a[0:3,3:4]


# In[71]:


a[0:5]


# In[72]:


a[3,4]


# In[73]:


a[3:4]


# In[2]:


a[0:4]


# In[ ]:





# In[75]:


a[0:1]


# In[7]:


a[0:][3][4]


# In[8]:


a[3][4]


# In[9]:


a[0:3,4]


# In[10]:


a[0:4]


# In[11]:


a[0,3]


# In[12]:


a[0,3,4]


# In[13]:


a[0],[3],[4]


# In[14]:


a[2],[5]


# In[15]:


a[2,5]


# In[16]:


a[0:3,4]


# In[17]:


a[0:4,3]


# In[18]:


a[0:0,5]


# In[19]:


a[0:0,3]


# In[20]:


a[0:0,4]


# In[21]:


a[0:0:4]


# In[22]:


a[0,3:4]


# In[26]:


a[0,3:5]


# In[47]:


import numpy as np
a=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30]])
a


# In[50]:


a[0,3:5]


# In[49]:


a[2:4,0:2]


# In[44]:


a[4:6,3:5]


# In[46]:


a[(0,1,2,3),(1,2,3,4)]


# In[7]:


import numpy as np
a=np.ones((2,3))
b=np. full((3,2),5)
print(a)
print(b)
out=np.matmul(a,b)
out


# In[9]:


a=np.array([[1,2,3],[4,5,6],[7,8,9]])
a


# In[10]:


np.max(a)


# In[11]:


np.min(a)


# In[12]:


np.max(a,axis=1)


# In[13]:


np.max(a,axis=0)


# In[14]:


a=np.array([1,2])
b=np.array([3,4])
print(a)
print(b)
np.sum([1,2,3])


# In[15]:


np.sum([a,b])


# In[16]:


np.sum(a)


# In[17]:


np.sum(b)


# In[18]:


np.sum([a,b],axis=0)


# In[19]:


np.sum([a,b],axis=1)


# In[20]:


a.sum(axis=0)


# In[22]:


import os 
os.getcwd()


# In[2]:


import numpy as np
data=np.genfromtxt('abc.txt',delimiter=',')
data


# In[4]:


a=data.astype('int32')
a


# In[5]:


a>5


# In[6]:


a<5


# In[7]:


a==5


# In[8]:


a<3


# In[9]:


np.all(a>5,axis=1)


# In[4]:


import os 
os.getcwd()


# In[6]:


import numpy as np
data=np.genfromtxt('aydin.txt',delimiter='')
data


# In[8]:


a=data.astype('int32')


# In[9]:


a>5


# In[10]:


a==5


# In[11]:


a<5


# In[12]:


np.all(a>5,axis=1)


# In[15]:


np.any(a<5,axis=1)
a


# In[33]:


import numpy as np
a=np.arange(2,11).reshape(3,3)
a


# In[34]:


a=np.full((3,3),(2))


# In[35]:


a=np.arange(12,39)
a


# In[37]:


a=np.array([[1,2,10],[4,5,6]])
a


# In[38]:


a.dtype


# In[39]:


a.type


# In[40]:


a.nbytes


# In[41]:


a.shpe


# In[42]:


a.shape


# In[1]:


import numpy as np
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
b=np.array([[2,3,4]])
print(b)


# In[2]:


np.sum((a,b))


# In[5]:


np.subtract(a,b)


# In[11]:


a=np.arange(2,11).reshape(3,3)
a


# In[12]:


a=np.array([1,1,2,2,3,4,5,6])
u=np.unique(a)
u


# In[14]:


a=np.array([1,1,1,2,2,3,3,4])
u,c=np.unique(a,return_counts=True)
print(u)
print(c)


# In[16]:


import numpy as np
a=np.array([1,1,1,3,4,5,5,5,6,7,7])
b=np.array
a


# In[4]:


import numpy as np
a=np.array([[4,0,0,0],[0,5,0,0],[0,0,6,0],[0,0,0,7]])
a


# In[5]:


np.diagflat([4,5,6,7])


# In[11]:


a=np.zeros(6)
a[5]=9
a


# In[12]:


a=np.arange(12,40)
a


# In[13]:


a=np.array([5,7,1,3])
np.flip(a)
a


# In[4]:


import numpy as np
a=np.array([10,10,20,30,30])
b=np.array([0,40])
a.put([0,4],b)
a


# In[2]:


import pandas as pd
a=pd.Series()
a


# In[1]:


import pandas as pd
import numpy as np
data=np.array([1,2,3])
a=pd.Series(data)
a


# In[ ]:





# In[ ]:




