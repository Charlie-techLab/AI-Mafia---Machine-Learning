#!/usr/bin/env python
# coding: utf-8

# In[1]:


name = input("Enter your name: ")
print(name)


# In[1]:


print("Hello World!")


# In[3]:


get_ipython().system('pip3 install numpy')


# In[10]:


##We are going to learn numpy now!


# In[11]:


'''
If you are not working on anaconda
then you can directly
pip3 install jupyterlab
'''


# In[3]:


import numpy as np


# In[4]:


# creating an array object
arr = np.array([[1, 2, 3],
               [4, 2, 5]])


# In[5]:


type(arr)


# In[6]:


arr.ndim


# In[7]:


#shape means number of rows and number of columns
arr.shape


# In[12]:


#In this case, shape means the number of elements which is 3
arr = np.array([1, 2, 3])


# In[13]:


arr.shape


# In[14]:


# creating array from list with type float
a = np.array([[1, 2, 4], [5, 8, 7]], dtype = 'float')
print("Array created using passed list:\n", a)


# In[15]:


# creating array from tuple
b = np.array((1, 3, 2))
print("Array created using passed tuple:\n", b)


# In[16]:


# creating a 3X4 array with all zeros
c = np.zeros((3, 4))
print("An array created initialized with all zeros:\n", c)


# In[17]:


# creating a 3X4 array with all ones
d = np.ones((3, 4))
print("An array created initialized with all ones:\n", d)


# In[21]:


# create an identity matrix
e = np.eye(3)
print("An array created initialized with an identity matrix:\n", e)


# In[19]:


#to find documentation we use ?
get_ipython().run_line_magic('pinfo', 'np.arange')


# In[20]:


# create a constant value array of complex type
d = np.full((3, 3), 6, dtype = 'complex')
print("An array initialized with all 6s. Array type is complex:\n", d)


# In[22]:


# create an array with random values
f = np.random.random((2, 2))
print("A random array:\n", f)


# In[26]:


# create a sequence of integers from 0 to 30 with steps of 5
g = np.arange(0, 30, 5)
print("A sequential array with steps of 5:\n", g)


# In[27]:


# create a sequence of 10 values in range 0 to 5 linspace = equal space
h = np.linspace(0, 5, 10)
print("A sequential array with 10 values between 0 and 5:\n", h)


# In[28]:


# reshaping 3x4 array to 2x2x3 array
arr = np.array([[1, 2, 3, 4],
              [5, 2, 4, 2],
              [1, 2, 0, 1]])
# 2 arrays, 2 rows, 3 columns
newarr = arr.reshape(2, 2, 3)
print("Original array:\n", arr)
print("Reshaped array:\n", newarr)


# In[42]:


# flatten array - converts th earray into a single object
arr = np.array([[1, 2, 3], [4, 5, 6]])
flarr = arr.flatten()
print("Original array:\n", arr)
print("Flattened array:\n", flarr)


# In[41]:


# reshaping 3x4 array to 2x2x3 array
arr = np.array([[1, 2, 3, 4],
              [5, 2, 4, 2],
              [1, 2, 0, 1]])
# 2 arrays, 2 rows, 3 columns
# the reshaping process can only with multiples of the same size of the
# number of elements. For example, we can do 4, 3; 3, 4; 6 x 2; 2,6
# because this multiplication is equal to 12 whic corresponds to the total
# number of elements in the array
newarr = arr.reshape(4, 3)
print("Original array:\n", arr)
print("Reshaped array:\n", newarr)


# In[51]:


#Slicing in Numpy

# arr[rows-> start:end+1:step , columns-> start:end+1:step]

# an exemplar array

# columns         0  1  2  3      rows
arr = np.array([[-1, 2, 0, 4],  # 0
              [4, -0.5, 6, 0],  # 1
              [2.6, 0, 7, 8],   # 2
              [3, -7, 4, 2.0]]) # 3
# slicing
#                       2nd row till end -> 2nd column till 3rd
# arr[rows, columns] -> [1: ,               :3]
temp = arr[1:, :3]
print("Array with second row till end and second column till 3rd:\n", temp)


# In[53]:


# another example
arr = np.array([[-1, 2, 0, 4],  # 0
              [4, -0.5, 6, 0],  # 1
              [2.6, 0, 7, 8],   # 2
              [3, -7, 4, 2.0]]) # 3
# slicing
#                       2nd row till end -> 2nd column till 3rd with 2 steps
# arr[rows, columns] -> [1: ,               :3]
temp = arr[1:, :3:2]
print("Array with second row till end and second column till 3rd with 2 steps:\n", temp)


# In[54]:


# one more example
arr = np.array([[-1, 2, 0, 4],  # 0
              [4, -0.5, 6, 0],  # 1
              [2.6, 0, 7, 8],   # 2
              [3, -7, 4, 2.0]]) # 3
# slicing
#                       2nd row till end -> 2nd column till 3rd with 2 steps
# arr[rows, columns] -> [1: ,               :3]
temp = arr[1::2, :3:2]
print("Array with second row till end with 2 steps and second column till 3rd with 2 steps:\n", temp)


# In[59]:


#                 0  1  2  3
arr = np.array([[-1, 2, 0, 4],  # 0
              [4, -0.5, 6, 0],  # 1
              [2.6, 0, 7, 8],   # 2
              [3, -7, 4, 2.0]]) # 3
# slicing
temp = arr[:2, ::2]
print("Array with first 2 rows and alternate columns (0 and 2):\n", temp)

# integer array indexing example
temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]]
print("Elements at indices (0, 3), (1, 2), (2, 1), (3, 0):\n", temp)

# boolean array indexing example
cond = arr > 0 # cond is a boolean array
temp = arr[cond]
print("Elements greater than 0:\n", temp)



# In[62]:


# Basic operations - this is the most important feature of Numpy
# the broadcast operations means that apply the same calculation to
# every and each one of the elements
a = np.array([1, 2, 5, 3])

# add 1 to every element
print("Adding 1 to every element: ", a+1)

# substract 3 from each element
print("Substracting 3 from each element:", a-3)

# multiply each element by 10
print("Multiplying each element by 10:", a*10)

# square each element
print("Squaring each element: ", a **2)

# modify existing array
a *= 2
print("Doubled each element of original array:", a)

# transpose of array
a = np.array([[1, 2, 3], [3, 4, 5], [9, 6, 0]])
print("Original array:\n", a)
print("Transpose of array:\n", a.T)


# In[71]:


# Unary operators
arr = np.array([[1, 5, 6],  
              [4, 7, 2],  
              [3, 1, 9]])

# maximum element of array
print("Largest element is:", arr.max())
print("Row-wise maximum elements: ", arr.max(axis = 1))

# minimum element of array
print("Shortest element is:", arr.min())
print("Column-wise minimum elements: ", arr.min(axis = 0))

# sum of array elements
print("Sum of all array elements:", arr.sum())

# cumulative sum along each row
print("Cumulative sum along each row:\n", arr.cumsum(axis = 1))

# cumulative sum along each column
print("Cumulative sum along each column:\n", arr.cumsum(axis = 0))


# In[72]:


a = np.array([[1, 2],
             [3, 4]])
b = np.array([[4, 3],
             [2, 1]])

# add arrays
print("Array sum:\n", a+b)

# multiply arrays (elementwise multiplication)
print("Array multiplication:\n", a*b)

# matrix multiplication
print("Matrix multiplication:\n", a.dot(b))


# In[73]:


# create an array of nine values
a = np.array([0, np.pi/2, np.pi])
print("Sin0 values of array elements:", np.sin(a))

# exponential values
a = np.array([0, 1, 2, 3])
print("Exponent of array elements:", np.exp(a))

# square root of array values
print("Square root of array elements:", np.sqrt(a))


# In[84]:


# Sorting array

a = np.array([[1, 4, 2],  
              [3, 4, 6],  
              [0, -1, 5]])

# sorted array
print("Array elements in sorted order:\n", np.sort(a, axis = None))

# sort array row-wise
print("Row-wise sorted array:\n", np.sort(a, axis = 1))

# sort array column-wise
print("Column-wise sorted array:\n", np.sort(a, axis = 0))

# specify sort algorithm
print("Column wise sort by applying merge-sort:\n", np.sort(a, axis = 0, kind = 'mergesort'))

# example to show sorting of structured array
## set alias names for dtypes
dtypes = [('name', 'S10'), ('grad_year', int), ('cgpa', float)]
# values to be put in array
values = [('Hrithik', 2009, 8.5), ('Ajay', 2008, 8.7), ('Pankaj', 2008, 7.9), ('Aakash', 2009, 9)]
# creating array
arr = np.array(values, dtype = dtypes)
print("Array sorted by names:\n", np.sort(arr, order = 'name'))
print("Array sorted by graduation year and then cgpa:\n", np.sort(arr, order = ['grad_year', 'cgpa']))


# In[87]:


# Stacking and Splitting
# Several arrays can be stacked together along different axes.
# np.vstack: To stack arrays along horizontal axis.
# np.column_stack: To stack 1-D arrays and columns into 2-D arrays.
# np.concatenate: To stack arrays along specified axis (axis is passed as argument).

a = np.array([[1, 2],
             [3, 4]])

b = np.array([[5, 6],
             [7, 8]])

# vertical stacking
print("Vertical stacking:\n", np.vstack((a, b)))

# horizontal stacking
print("Horizontal stacking:\n", np.hstack((a, b)))

c = [5, 6]

# stacking columns
print("Column stacking:\n", np.column_stack((a, c)))

# concatenation method
print("Concatenating to 2nd axis:\n", np.concatenate((a, b), 1))



# In[88]:


# For splitting we have these functions:
# np.hsplit: Split array along horizontal axis.
# np.vsplit: Split array along vertical axis.
# np.array_split: Split array along specified axis.

a = np.array([[1, 3, 5, 7, 9, 11],
             [2, 4, 6, 8, 10, 12]])

# horizontal splitting
print("Splitting along horizontal axis into 2 parts:\n", np.hsplit(a, 2))

# vertical splitting
print("Splitting along vertical axis into 2 parts:\n", np.vsplit(a, 2))



# In[89]:


# Some more Numpy Functions - Statistics
# min, max
# mean
# median
# average
# variance
# standard deviation

a = np.array([[1, 2, 3, 4], [7, 6, 2, 0]])
print(a)
print(np.min(a))
# Specify axis for the direction in case of multidim array
print(np.min(a, axis=0))
print(np.min(a, axis=1))



# In[90]:


b = np.array([1, 2, 3, 4, 5])
m = sum(b)/5
print(m)

print(np.mean(b))
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))



# In[91]:


c = np.array([1, 5, 4, 2, 0])
print(np.median(c))

# Mean vs Average is Weighted
print(np.mean(c))

# Weights
w = np.array([1, 1, 1, 1, 1])
print(np.average(c, weights=w))

# weighted mean => n1*w1 + n2*w2 / n1+n2

# Standard Deviation
u = np.mean(c)
myStd = np.sqrt(np.mean(abs(c-u)**2))
print(myStd)

# Inbuilt Function
dev = np.std(c)
print(dev)

# Variance
print(myStd**2)
print(np.var(c))










# In[93]:


# Numpy Random Module
# rand: Random values in a given shape.
# randn: Return a sample (or samples) from the "standard normal" distribution.
# randint: Return random integers from low (inclusive) to high (exclusive).
# random: Return random floats in the half-open interval [0.0, 1.0].
# choice: Generates a random sample from a given 1-D array.
# shuffle: Shuffles the contents of a sequence.

a = np.arange(10) + 5
print(a)

np.random.seed(1)
np.random.shuffle(a)
print(a)

# Returns values from a Standard Normal Distribution
a = np.random.randn(2, 3)
print(a)

a = np.random.randint(5, 10, 3)
print(a)

# Randomly pick one element from an array
element = np.random.choice([1, 4, 3, 2, 11, 27])
print(element)


# In[ ]:
