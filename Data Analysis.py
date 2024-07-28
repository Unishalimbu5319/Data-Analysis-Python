#!/usr/bin/env python
# coding: utf-8

# In[180]:


#importing the libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[183]:


#to load cv file in dataframe 
dataframe = pd.read_csv('DataScienceSalarie.csv')
dataframe


# In[184]:


#to load data into pandas DataFrame
df = pd.DataFrame(dataframe)

print(df)


# In[190]:


df.info()


# In[185]:


# remove unnecessary columns
df.drop(columns=['salary', 'salary_currency'])


# In[186]:


#to add NaN value in dataframe
row_index = 0
column_name = 'experience_level', 'employment_type', 'job_title' 
df.at[row_index, column_name] = np.nan
print("NaN value added:")
df


# In[187]:


#to remove the NaN missing values from updated dataframe 
rows_before, cols_before = df. shape
df = df.dropna()
rows_after, cols_after = df.shape
nan_dropped = (rows_before - rows_after)
print("number of NaN values dropped:", nan_dropped)
df


# In[179]:


#to find the total duplicated value amount
duplicate_values = df[df.duplicated()]
duplicate_value_columns = df[df.duplicated()]
total_value = df.duplicated().sum()

print("Duplicate rows based on all columns:")
print(duplicate_values)
print("Duplicate Rows base on workyear and job title")
print(duplicate_value_columns)
print("Total duplicated values are:", total_value)


# In[52]:


for column in df.columns:
    unique_values = df[column].unique()
    print("Unique values in column are'{}': {}".format(column, unique_values))


# In[60]:


#Replacing the value of experience level

level_mapping = {
    'SE' : 'Senior Level/Expert',
    'MI' : 'Medium Level/Intermediate',
    'EN' : 'Entry Level',
    'EX' : 'Executive Level'
}

df.loc['experience_level'] = df['experience_level'].replace(level_mapping)
df


# In[61]:


df.describe()


# In[83]:


#summary statistics of sum, mean, standard deviation, and kurtosis of any chosen variable
sum = df['salary'].sum()
print("Sum:", sum)
mean = df['salary'].mean()
print("Mean:", mean)
std_deviation = df['salary'].std()
print("Salary:", std_deviation)
Skewness = df['salary'].skew()
print("Skewness:", Skewness)
Kurtosis = df['salary'].kurt()
print("Kurtosis:",Kurtosis)


# In[82]:


#to calculate and show correlation of all variables

df_categorical = df.select_dtypes(include = 'category')
df_categorical = df_categorical.apply(lambda x: x.cat.codes)
df_integer = df.select_dtypes(include = ['int', 'float'])

df_combined = pd.concat([df_numeric, df_categorical_codes], axis = 1)

correlation_matrix = df_combined.corr()

print("Correlation of all variables")
print(correlation_matrix)



# In[94]:


plt.figure(figsize = (10,8))
sns.set_theme(style="whitegrid")
heatmap = sns.heatmap(data=correlation_matrix, cmap="Purples", fmt='.2g')


# In[71]:


df.info()


# In[78]:


# df['experience_len'] = df['experience_level'].apply(len)
# df['employment_len'] = df['employment_type'].apply(len)
# df['work_year'] = df['work_year'].astype('string')
# df['job_len'] = df['job_title'].apply(len)
# df['salary_len'] = df['salary_currency'].apply(len)
# df['employee_len'] = df['employee_residence'].apply(len)
# df['company_loc_len'] = df['company_location'].apply(len)
# df['company_size_len'] = df['company_size'].apply(len)
# correlation = df[['experience_len', 'employment_len', 'work_year', 'job_len', 'salary_len',
# 'employee_len', 'company_loc_len', 'company_size_len']].corr()
# correlation


# In[102]:


#top 15 jobs 
top_jobs = df.groupby('job_title', observed = False)['salary_in_usd'].sum()
top_jobs = top_jobs.sort_values(ascending = False).head(15)
top_jobs = top_jobs.reset_index()
top_jobs


# In[104]:


plt.figure(figsize = (10, 6))
plt.bar(top_jobs['job_title'], top_jobs['salary_in_usd'], color = 'purple')
plt.xlabel('Job Title')
plt.ylabel('Total Salary (USD)')
plt.title('Top 15 Jobs by salary ')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show


# In[129]:


# to find the highest paying job 
highest_paying = df.groupby(df['job_title'],observed=False)['salary_in_usd'].sum()
highest_paying_job = highest_paying.idxmax()
highest_paying_salary = highest_paying.max()

print("Highest salary",highest_paying_salary)
print("Highest paying job", highest_paying_job)



# In[132]:


#to plot bargraph
top_10_paying = highest_paying.nlargest(10) #only selcting top 10
highest_paying_job = top_10_paying.idxmax()
highest_paying_salary = top_10_paying.max()

plt.figure(figsize = (15,8))
plt.bar(top_10_paying.index, top_10_paying.values, color="blue")
plt.xlabel('Job Title')
plt.ylabel('Total Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[135]:


#Plotting the salaries by experience
salaries_experience = df.groupby('experience_level',observed=False)['salary_in_usd'].mean()
plt.figure(figsize=(10,6))
salaries_experience.plot(kind='bar', color = 'red')
plt.xlabel('Experience Level')
plt.ylabel('Salary USD')
plt.title('Salary by Experince')
plt.xticks(rotation=45)
plt.show()


# In[158]:


plt.figure(figsize = (8,6))
plt.hist(df['experience_level'], bins = range(len(df['experience_level'].unique())+1), color='purple', edgecolor='white')
plt.xlabel('Experience Level')
plt.ylabel('Frequency')
plt.title('Histogram of Experience Level')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


# In[189]:


#boxplot of comanysize
warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
sns.boxplot(x='company_size', y='salary_in_usd', data=df)
plt.xlabel('Company Size')
plt.ylabel('Salary')
plt.title('Box Plot of Salary by Company Size')
plt.tight_layout()
plt.show()


# In[ ]:




