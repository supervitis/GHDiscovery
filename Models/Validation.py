
# coding: utf-8

# In[2]:

import sys
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib
from threading import Thread
from queue import Queue
import math
import requests
import time 
from datetime import timedelta
import pickle
from datetime import datetime


# In[5]:

df = pd.read_csv('Voted/backbone100.csv', index_col=0, encoding="utf-8", delimiter=";")


# In[20]:

df


# In[34]:

df = pd.read_csv('Voted/backbone100.csv', index_col=0, encoding="utf-8", delimiter=";")
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('test', 'Tests')
df[['EREN']] = df[['EREN']].replace('Test', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('doc', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('docs', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('documentation', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('src', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('core', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('development', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build ', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('compile', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('tools', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('merge', 'Build')
df.head()


# In[63]:

df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN != df.SANJA), df.EREN, df.vote)
df['vote'] = np.where((df.SANJA == df.MIRELLA) & (df.EREN != df.SANJA), df.SANJA, df.vote)
df['vote'] = np.where((df.SANJA == df.EREN) & (df.EREN != df.MIRELLA), df.SANJA, df.vote)
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN == df.SANJA), df.EREN, df.vote)
df['check'] = np.where((df.EREN != df.MIRELLA) & (df.EREN != df.SANJA) & (df.MIRELLA != df.SANJA),"true","false" )
for i in range(1,10):
    classCount = df.check.value_counts()
print(classCount)


# In[67]:

def accuracy_dist(name, x, y, df, dfX, dfY):
    print("----")
    print(name)
    correct = len(df[dfX.isin([x])][dfY.isin([y])])
    wrong = len(df[dfX.isin([x])][~dfY.isin([y])]) + len(df[~dfX.isin([x])][dfY.isin([y])])
    print(correct/(correct + wrong))
    print("Same")
    print(correct)
    print("Different")
    print(wrong)
    print("----")
print("AGAINST FIRST MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted, df.vote)
print("AGAINST SECOND MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted2, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted2, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted2, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted2, df.vote)


# In[70]:




# In[77]:

df = pd.read_csv('Voted/bootstrap100.csv', index_col=0, encoding="utf-8", delimiter=",")
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('test', 'Tests')
df[['EREN']] = df[['EREN']].replace('Test', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('doc', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('docs', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('documentation', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('src', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('core', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('development', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('develop', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build ', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('compile', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('tools', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('merge', 'Build')
df.head()


# In[78]:

df['vote'] = "false"
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN != df.SANJA), df.EREN, df.vote)
df['vote'] = np.where((df.SANJA == df.MIRELLA) & (df.EREN != df.SANJA), df.SANJA, df.vote)
df['vote'] = np.where((df.SANJA == df.EREN) & (df.EREN != df.MIRELLA), df.SANJA, df.vote)
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN == df.SANJA), df.EREN, df.vote)
df['check'] = np.where((df.EREN != df.MIRELLA) & (df.EREN != df.SANJA) & (df.MIRELLA != df.SANJA),"true","false" )
for i in range(1,10):
    classCount = df.check.value_counts()
print(classCount)


# In[84]:

for i in range(1,10):
    classCount = df.EREN.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.MIRELLA.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.SANJA.value_counts()
print(classCount)


# In[79]:

df


# In[80]:

def accuracy_dist(name, x, y, df, dfX, dfY):
    print("----")
    print(name)
    correct = len(df[dfX.isin([x])][dfY.isin([y])])
    wrong = len(df[dfX.isin([x])][~dfY.isin([y])]) + len(df[~dfX.isin([x])][dfY.isin([y])])
    print(correct/(correct + wrong))
    print("Same")
    print(correct)
    print("Different")
    print(wrong)
    print("----")
print("AGAINST FIRST MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted, df.vote)
print("AGAINST SECOND MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted2, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted2, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted2, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted2, df.vote)


# In[85]:

df = pd.read_csv('Voted/codemirror100.csv', index_col=0, encoding="utf-8", delimiter=",")
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('test', 'Tests')
df[['EREN']] = df[['EREN']].replace('Test', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('doc', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('docs', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('documentation', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('src', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('core', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('development', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('develop', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build ', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('compile', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('tools', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('merge', 'Build')
df.head()


# In[86]:

for i in range(1,10):
    classCount = df.EREN.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.MIRELLA.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.SANJA.value_counts()
print(classCount)


# In[87]:

df['vote'] = "false"
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN != df.SANJA), df.EREN, df.vote)
df['vote'] = np.where((df.SANJA == df.MIRELLA) & (df.EREN != df.SANJA), df.SANJA, df.vote)
df['vote'] = np.where((df.SANJA == df.EREN) & (df.EREN != df.MIRELLA), df.SANJA, df.vote)
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN == df.SANJA), df.EREN, df.vote)
df['check'] = np.where((df.EREN != df.MIRELLA) & (df.EREN != df.SANJA) & (df.MIRELLA != df.SANJA),"true","false" )
for i in range(1,10):
    classCount = df.check.value_counts()
print(classCount)


# In[88]:

def accuracy_dist(name, x, y, df, dfX, dfY):
    print("----")
    print(name)
    correct = len(df[dfX.isin([x])][dfY.isin([y])])
    wrong = len(df[dfX.isin([x])][~dfY.isin([y])]) + len(df[~dfX.isin([x])][dfY.isin([y])])
    print(correct/(correct + wrong))
    print("Same")
    print(correct)
    print("Different")
    print(wrong)
    print("----")
print("AGAINST FIRST MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted, df.vote)
print("AGAINST SECOND MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted2, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted2, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted2, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted2, df.vote)


# In[91]:

df = pd.read_csv('Voted/less100.csv', index_col=0, encoding="utf-8", delimiter=",")
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('test', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('Test', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('Test ', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('doc', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('docs', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('documentation', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('src', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('core', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('development', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('develop', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build ', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('compile', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('tools', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('merge', 'Build')
for i in range(1,10):
    classCount = df.EREN.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.MIRELLA.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.SANJA.value_counts()
print(classCount)


# In[92]:

df['vote'] = "false"
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN != df.SANJA), df.EREN, df.vote)
df['vote'] = np.where((df.SANJA == df.MIRELLA) & (df.EREN != df.SANJA), df.SANJA, df.vote)
df['vote'] = np.where((df.SANJA == df.EREN) & (df.EREN != df.MIRELLA), df.SANJA, df.vote)
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN == df.SANJA), df.EREN, df.vote)
df['check'] = np.where((df.EREN != df.MIRELLA) & (df.EREN != df.SANJA) & (df.MIRELLA != df.SANJA),"true","false" )
for i in range(1,10):
    classCount = df.check.value_counts()
print(classCount)


# In[93]:

def accuracy_dist(name, x, y, df, dfX, dfY):
    print("----")
    print(name)
    correct = len(df[dfX.isin([x])][dfY.isin([y])])
    wrong = len(df[dfX.isin([x])][~dfY.isin([y])]) + len(df[~dfX.isin([x])][dfY.isin([y])])
    print(correct/(correct + wrong))
    print("Same")
    print(correct)
    print("Different")
    print(wrong)
    print("----")
print("AGAINST FIRST MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted, df.vote)
print("AGAINST SECOND MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted2, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted2, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted2, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted2, df.vote)


# In[94]:

df = pd.read_csv('Voted/reveal100.csv', index_col=0, encoding="utf-8", delimiter=",")
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('test', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('Test', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('Test ', 'Tests')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('doc', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('docs', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('documentation', 'Docs')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('src', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('core', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('development', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('develop', 'Core')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('build ', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('compile', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('tools', 'Build')
df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']] = df[['predicted2', 'EREN', 'MIRELLA', 'SANJA']].replace('merge', 'Build')
for i in range(1,10):
    classCount = df.EREN.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.MIRELLA.value_counts()
print(classCount)
for i in range(1,10):
    classCount = df.SANJA.value_counts()
print(classCount)


# In[95]:

df['vote'] = "false"
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN != df.SANJA), df.EREN, df.vote)
df['vote'] = np.where((df.SANJA == df.MIRELLA) & (df.EREN != df.SANJA), df.SANJA, df.vote)
df['vote'] = np.where((df.SANJA == df.EREN) & (df.EREN != df.MIRELLA), df.SANJA, df.vote)
df['vote'] = np.where((df.EREN == df.MIRELLA) & (df.EREN == df.SANJA), df.EREN, df.vote)
df['check'] = np.where((df.EREN != df.MIRELLA) & (df.EREN != df.SANJA) & (df.MIRELLA != df.SANJA),"true","false" )
for i in range(1,10):
    classCount = df.check.value_counts()
print(classCount)


# In[96]:

def accuracy_dist(name, x, y, df, dfX, dfY):
    print("----")
    print(name)
    correct = len(df[dfX.isin([x])][dfY.isin([y])])
    wrong = len(df[dfX.isin([x])][~dfY.isin([y])]) + len(df[~dfX.isin([x])][dfY.isin([y])])
    print(correct/(correct + wrong))
    print("Same")
    print(correct)
    print("Different")
    print(wrong)
    print("----")
print("AGAINST FIRST MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted, df.vote)
print("AGAINST SECOND MODEL")
accuracy_dist("CORE",  "Core", "Core", df, df.predicted2, df.vote)
accuracy_dist("BUILD", "Build", "Build", df, df.predicted2, df.vote)
accuracy_dist("DOCS",  "Docs", "Docs", df, df.predicted2, df.vote)
accuracy_dist("TEST", "Tests", "Tests",  df, df.predicted2, df.vote)


# In[ ]:



