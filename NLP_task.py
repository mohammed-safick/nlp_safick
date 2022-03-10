#!/usr/bin/env python
# coding: utf-8

# In[123]:


#installing packages

# 1. pandas library installation
# these packages has been to be run either in command prompt or anaconda prompt 
# type "command prompt" in search option and type "pip install pandas"
# or if you have anaconda prompt installed, type "anaconda prompt" in search option and then
# For installing pandas packages in anaconda ==> "conda install pandas"

# 2. scikit-learn library installation
# these packages has been to be run either in command prompt or anaconda prompt 
# type "command prompt" in search option and type "pip install -U scikit-learn"
# or if you have anaconda prompt installed, type "anaconda prompt" in search option and then
# For installing scikit-learn packages in anaconda ==> "conda install scikit-learn"

import pandas as pd

indept_df = pd.read_csv(r"C:\Users\safic\Desktop\peers\input\Learning Catalogue 2.csv") #input file
dept_df = pd.read_excel(r'C:\Users\safic\Desktop\peers\input\Sample_Skills.xlsx',sheet_name='Sheet1',engine='openpyxl') #input file


# In[124]:


# using dropna() function we will be removing the empty or (NaN) rows and columns in the DataFrame

dept_df = dept_df.dropna() 


# In[125]:


from sklearn.metrics.pairwise import cosine_similarity # cosine similarity is a metric used to measure the similarity of two vectors
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer is a numerical statistic that measures the importance of the word in a document
tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2)) # word embedding is done by converting each word into numbers


# In[126]:


Matched_Skill = pd.DataFrame()
for i in range(0,len(indept_df)):
    dff = indept_df.filter(items = [i], axis=0)
    desc_vectorizer = tfidf.fit_transform(dff['Description']).toarray() # vectorising the description part
    skill_vectorizer = tfidf.transform(dept_df['Skill']).toarray() # vectorising the skill part
    
    df2 = dept_df.copy()
    df2['Similarity_Score'] = cosine_similarity(skill_vectorizer,desc_vectorizer) # occuring the similarity score by using the cosine similarity
    
    df2 = df2.sort_values(by = 'Similarity_Score', ascending = False).reset_index(drop=True) # to have the score from the descending order
    
    df2 = df2.head(5) 
    
    M_Skill = pd.DataFrame()
    M_Skill['Skill'] = df2['Skill']
    M_Skill['Similarity_Score'] = df2['Similarity_Score']
    M_Skill['Title'] = dff['Title'][i]
    M_Skill['Description'] = dff['Description'][i]

    Matched_Skill = Matched_Skill.append(M_Skill,ignore_index=True) 
    


# In[127]:


Matched_Skill.to_csv(r'C:\Users\safic\Desktop\peers\output\matched_skill.csv') # output file






