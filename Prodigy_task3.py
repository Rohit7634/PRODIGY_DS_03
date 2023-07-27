#!/usr/bin/env python
# coding: utf-8

# # Importin libraries

# In[62]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the dataset

# In[79]:



df = pd.read_csv("bankpp.csv")


# ## Print data head

# In[63]:


df.head()


# ## Print data tail

# In[64]:


df.tail(8)


# ## Printing number of columns and rows

# In[65]:


df.shape


# In[66]:


print("Number of rows",df.shape[0])
print("Number of columns",df.shape[1])


# ## Printing data information and its type

# In[67]:


df.info()


# ## Checking missing value or null values in dataset

# In[69]:


print("any Missing Value? ", df.isnull().values.any())


# In[70]:


df.isnull().sum()


# In[71]:


sns.heatmap(df.isnull())


# In[72]:


per=df.isnull().sum()*100/len(df)


# In[73]:


print(per)


# ## Checking Duplicate value in dataset

# In[75]:


dup=df.duplicated().any()


# In[76]:


print(dup)


# ## Find all numerical value mean median and more

# In[77]:


df.describe()


# In[78]:


df.describe(include='all')


# ## Convert categorical variables into numerical labels

# In[29]:



df['marital'] = df['marital'].map({'single': 1, 'married': 2, 'divorced': 3})
df['education'] = df['education'].map({'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 4})
df['default'] = df['default'].map({'no': 0, 'yes': 1})
df['housing'] = df['housing'].map({'no': 0, 'yes': 1})
df['loan'] = df['loan'].map({'no': 0, 'yes': 1})
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Drop unnecessary columns
df = df.drop(['job', 'contact', 'month', 'day', 'poutcome'], axis=1)

# Split the data into features and target variable
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target variable


# ## Split the dataset into training and testing sets (70% train, 30% test)

# In[30]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ## Initialize the decision tree classifier

# In[80]:



clf = DecisionTreeClassifier()



# ## Train the decision tree classifier using the training data
# 

# In[81]:


clf.fit(X_train, y_train)


# ## Predict the target variable for the test data

# In[32]:



y_pred = clf.predict(X_test)


# ## Evaluate the model by calculating accuracy, precision, recall, and F1 score

# In[33]:


# Evaluate the model by calculating accuracy, precision, recall, and F1 score
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[40]:


df.hist()
plt.xticks(rotation=90)


# In[43]:


plt.hist(y_pred)


# In[50]:


plt.hist(X_train.head(10))


# In[51]:


plt.hist(X_train.tail(10))


# In[46]:


X_train.hist()


# In[49]:


plt.hist(X_test.head(10))


# In[53]:


plt.hist(X_test.tail(10))


# In[60]:


sns.heatmap(X_train)


# In[61]:


sns.heatmap(X_test)


# In[ ]:




