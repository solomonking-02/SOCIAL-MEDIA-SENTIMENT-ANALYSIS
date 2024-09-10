#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[23]:


df = pd.read_csv(r'C:\Users\shamh\Downloads\Housing (1).csv')
df.head()


# In[25]:


# missing values/null values checking
df.isna().sum()


# In[26]:


sns.heatmap(df.isna())


# In[27]:


# duplicated values checking
df.duplicated().sum()


# In[28]:


# information about dataset
df.info()


# In[29]:


# unique values identification
df.nunique()


# In[30]:


# converting categorical values to numerical values
cat_col = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']


# In[31]:


# Iterate through categorical columns (excluding the last one) in the DataFrame 'df'
for col in cat_col[:-1]:
    # Convert values to binary (1 for 'yes' and 0 for anything else) using a lambda function
    df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)


# In[32]:


df1 = df.copy()
df1


# In[33]:


# descriptive statistics 
df1.describe()


# In[34]:


# converting categorical to numerical columns
furnishing_mapping = {
    'furnished': 1,
    'semi-furnished': 0,
    'unfurnished': -1 
}


# In[35]:


# Use a lambda function with `apply` to convert the categorical data to numerical data
df1['furnishingstatus'] = df1['furnishingstatus'].apply(lambda x:furnishing_mapping[x])


# In[36]:


df1['furnishingstatus']


# In[37]:


# checking correlation between variables 
plt.figure(figsize=(18,6))
sns.heatmap(df1.corr(), cmap='coolwarm', annot=True)


# In[38]:


# Removing outliers using (Z-score)
from scipy import stats

z = np.abs(stats.zscore(df1))


# In[39]:


# z > 3 shows outliers it is thersold

print(np.where(z>3))


# In[40]:


# Removing

new_df = df1[(z<3).all (axis = 1)]  # New dataset without outliers


# In[41]:


new_df.head()


# In[42]:


# understanding variables with new data (without outliers)
# Numerical columns
num_col = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']


# In[43]:


for col in num_col:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
    
    # Histogram plot
    sns.histplot(new_df[col], ax=ax[0])
    ax[0].set_title(f'Histogram of {col}')
    
    # Box plot
    sns.boxplot(x=new_df[col], ax=ax[1])
    ax[1].set_title(f'Boxplot of {col}')
    
    plt.show()


# In[44]:


# categorical columns
cat_col = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']


# In[45]:


plt.figure(figsize=(20, 12))
for i in range(len(cat_col[:-1])):
    plt.subplot(3,2,i+1)
    sns.boxplot(x = new_df[cat_col[i]], y = new_df.price )
    plt.suptitle(f'{col.title()}',weight='bold')
plt.show()


# In[46]:


# train and test data 
X = new_df[['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating',
             'airconditioning','parking','prefarea','furnishingstatus']]   # Independent Features
y = new_df['price'] # Dependent Features


# In[47]:


# linear Model


# In[48]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[49]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[50]:


# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[51]:


from sklearn.linear_model import LinearRegression


# In[52]:


lm = LinearRegression()
lm.fit(X_train,y_train)
# make prediction using test data
y_pred = lm.predict(X_test)


# In[53]:


# model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[54]:


mse = mean_squared_error ( y_test, y_pred ) # Mean Squared Error
print(f"Mean Squared Error (MSE): {mse}")
mae = mean_absolute_error ( y_test, y_pred )
print(f"Mean Absolute Error (MAE): {mae}")
r2 = r2_score ( y_test, y_pred ) # R2 
print(f"R-squared (R2): {r2}")


# In[55]:


# prediction plot using scatter plot

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


# In[56]:


# Manually input features for prediction

new_data = np.array([[9100,4,1,2,1,1,1,0,1,2,1,1]]) # Replace these values with your own features

# Make predictions on the new data
predicted_price = lm.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")


# In[ ]:




