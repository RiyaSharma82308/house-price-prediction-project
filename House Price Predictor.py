

import pandas as pd


housing=pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing['AGE'].value_counts()


# In[7]:


housing.describe()


# In[8]:


# %matplotlib inline  
# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))


# ## Train-Test Splitting

# In[9]:


# import numpy as np
# def split_train_data(data,test_ratio):
#     np.random.seed(42)
#     shuffle=np.random.permutation(len(data));
#     print(shuffle)
#     test_set_size=int(len(data)*test_ratio)
#     train=shuffle[test_set_size:]
#     test=shuffle[:test_set_size]
#     return data.iloc[train], data.iloc[test]


# In[10]:


# train_set,test_set=split_train_data(housing,0.2)


# In[11]:


# print(f" rows in train set are: {len(train_set)}\n rows in test set are: {len(test_set)}")


# In[12]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f" rows in train set are: {len(train_set)}\n rows in test set are: {len(test_set)}")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[14]:


strat_train_set['CHAS'].value_counts()


# In[15]:


housing=strat_train_set.copy()


# ## Looking for Correlations

# In[16]:


corr_matrix=housing.corr()


# In[17]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[19]:


housing.plot(kind='scatter', x='RM',y='MEDV',alpha=0.4)


# In[20]:


housing['TAXRM']=housing['TAX']/housing['RM']
housing.head()
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing.plot(kind='scatter', x='TAXRM',y='MEDV',alpha=0.4)


# In[21]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[22]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# ## To replace the missing values with median

# In[23]:


# from sklearn.impute import SimpleImputer
# imputer=SimpleImputer(strategy='median')
# imputer.fit(housing)
# imputer.statistics_
# X=imputer.transform(housing)
# housing_tr=pd.DataFrame(X,Coulumns=housing.columns)
# housing_tr.describe()


# ## Scikit learn design

# it has estimators, transformers & predictors

# ## Creating a Pipeline

# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([('std_scaler',StandardScaler()),])


# In[25]:


housing_num_tr=my_pipeline.fit_transform(housing)
housing_num_tr.shape


# ## Selecting a desired model for project 

# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[27]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)


# In[28]:


model.predict(prepared_data)


# In[29]:


list(some_labels)


# ## Evaluating the model

# In[30]:


import numpy as np
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[31]:


rmse      


# ## better evaluation technique- cross-validation

# In[32]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[33]:


rmse_scores


# In[34]:


def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Standard deviation: ",scores.std())


# In[35]:


print_scores(rmse_scores)


# ## Saving the model

# In[36]:


from joblib import dump,load
dump(model,'First_Project.joblib')


# ## Testing model on test-data

# In[37]:


X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
final_rmse


# ## Using the models

# In[40]:


model=load('First_Project.joblib')
# prepared_data
features=np.array([[ 0.15682292, -0.4898311 ,  0.98336806, -0.27288841,  0.47919371,
         0.28660512,  0.87020968, -0.68730678,  1.63579367,  1.50571521,
         0.81196637,  0.44624347,  0.81480158]])
model.predict(features)






