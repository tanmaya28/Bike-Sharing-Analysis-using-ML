#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm


# In[3]:


df = pd.read_csv("F:\Projects & Assignments\day.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.duplicated().sum()


# In[7]:


df.drop(['instant', 'dteday','casual','registered','atemp'], axis=1, inplace=True)


# In[8]:


df.shape


# In[9]:


cat_vars = ['season','yr','mnth','holiday','weekday', 'workingday','weathersit']
num_vars = ['temp', 'hum','windspeed','cnt']


# In[10]:


df[cat_vars] = df[cat_vars].astype('category')


# In[11]:


df.describe()


# In[12]:


df.describe(include=['category'])


# In[13]:


df['season'] = df['season'].map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})

df['weekday'] = df['weekday'].map({0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'})

df['mnth'] = df['mnth'].map({1:'jan', 2:'feb', 3:'mar', 4:'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct',
                             11: 'nov', 12:'dec'})
df['weathersit'] = df['weathersit'].map({1: 'Clear_FewClouds', 2: 'Mist_Cloudy', 3: 'LightSnow_LightRain', 4: 'HeavyRain_IcePallets'})


# In[14]:


df.info()


# In[15]:


plt.figure(figsize=(20,5))
plt.plot(df.cnt)
plt.show()


# In[16]:


var = df.select_dtypes(exclude = 'category').columns
col = 2
row = len(var)//col+1

plt.figure(figsize=(12,8))
plt.rc('font', size=12)
for i in list(enumerate(var)):
    plt.subplot(row, col, i[0]+1)
    sns.boxplot(df[i[1]])    
plt.tight_layout()   
plt.show()


# In[17]:


def percentage_outlier(x):
    iqr = df[x].quantile(0.75)-df[x].quantile(0.25)
    HL = df[x].quantile(0.75)+iqr*1.5
    LL = df[x].quantile(0.25)-iqr*1.5
    per_outlier = ((df[x]<LL).sum()+(df[x]>HL).sum())/len(df[x])*100
    per_outlier = round(per_outlier,2)
    return(per_outlier)

print('Percentage of outlier (hum): ', percentage_outlier('hum'))
print('Percentage of outlier (windspeed): ', percentage_outlier('windspeed'))


# In[18]:


df_piplot=df.select_dtypes(include='category')
plt.figure(figsize=(18,16))
plt.suptitle('pie distribution of categorical features', fontsize=20)
for i in range(1,df_piplot.shape[1]+1):
    plt.subplot(3,3,i)
    f=plt.gca()
    f.set_title(df_piplot.columns.values[i-1])
    values=df_piplot.iloc[:,i-1].value_counts(normalize=True).values
    index=df_piplot.iloc[:,i-1].value_counts(normalize=True).index
    plt.pie(values,labels=index,autopct='%1.0f%%')
plt.show()


# In[19]:


sns.pairplot(df.select_dtypes(['int64','float64']), diag_kind='kde')
plt.show()


# In[20]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[21]:


col = 3
row = len(cat_vars)//col+1

plt.figure(figsize=(15,12))
for i in list(enumerate(cat_vars)):
    plt.subplot(row,col,i[0]+1)
    sns.boxplot(x = i[1], y = 'cnt', data = df)
    plt.xticks(rotation = 90)
plt.tight_layout(pad = 1)    
plt.show()


# In[22]:


dummy_vars = pd.get_dummies(df[['season','weekday','mnth','weathersit']],drop_first=True)

df = pd.concat([df,dummy_vars], axis = 1)

df.drop(['season','weekday','mnth','weathersit'], axis=1, inplace=True)

df.head()


# In[23]:


df.shape


# In[25]:


df.info()


# In[26]:


df[['yr','holiday','workingday']]= df[['yr','holiday','workingday']].astype('uint8')
df.info()


# In[28]:


df[['yr','holiday','workingday']]= df[['yr','holiday','workingday']].astype('uint8')
df.info()


# In[29]:


df_train, df_test = train_test_split(df, train_size = 0.7, random_state = 10 )
print(df_train.shape)
print(df_test.shape)


# In[30]:


scaler = MinMaxScaler()
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()


# In[31]:


df_test.head()


# In[32]:


df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()


# In[33]:


y_train = df_train.pop('cnt')
X_train = df_train
X_train.head()


# In[34]:


y_test = df_test.pop('cnt')
X_test = df_test

X_test.head()


# In[35]:


X_train.columns


# In[37]:


lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)
col = X_train.columns[rfe.support_]
col


# In[38]:


X_train_rfe = X_train[col]


# In[39]:


def sm_linearmodel(X_train_sm):
   X_train_sm = sm.add_constant(X_train_sm)

   lm = sm.OLS(y_train,X_train_sm).fit()
   return lm


# In[40]:


def vif_calc(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    vif = vif.sort_values(by='VIF', ascending = False)
    return vif


# In[41]:


lm_1 = sm_linearmodel(X_train_rfe)
print(lm_1.summary())
print(vif_calc(X_train_rfe))


# In[42]:


pvalue = lm_1.pvalues
while(max(pvalue)>0.05):
    maxp_var = pvalue[pvalue == pvalue.max()].index
    print('Removed variable:' , maxp_var[0], '    P value: ', round(max(pvalue),3))
    
    X_train_rfe = X_train_rfe.drop(maxp_var, axis = 1)
    lm_1 = sm_linearmodel(X_train_rfe)
    pvalue = lm_1.pvalues


# In[43]:


print(lm_1.summary())

print(vif_calc(X_train_rfe))


# In[44]:


X_train_new = X_train_rfe.drop(['hum'],axis = 1)

lm_2 = sm_linearmodel(X_train_new)
print(lm_2.summary())
print(vif_calc(X_train_new))


# In[45]:


X_train_new = X_train_new.drop(['season_fall'],axis = 1)
lm_3 = sm_linearmodel(X_train_new)
print(lm_3.summary())
print(vif_calc(X_train_new))


# In[46]:


X_train_new = X_train_new.drop(['mnth_mar'],axis = 1)
lm_4 = sm_linearmodel(X_train_new)
print(lm_4.summary())
print(vif_calc(X_train_new))


# In[47]:


X_train_new = X_train_new.drop(['mnth_oct'],axis = 1)
lm_5 = sm_linearmodel(X_train_new)
print(lm_5.summary())
print(vif_calc(X_train_new))


# In[48]:


lm_final = lm_5
var_final = list(lm_final.params.index)
var_final.remove('const')
print('Final Selected Variables:', var_final)
print('\033[1m{:10s}\033[0m'.format('\nCoefficent for the variables are:'))
print(round(lm_final.params,3))


# In[49]:


X_train_res = X_train[var_final]


# In[50]:


X_train_res = sm.add_constant(X_train_res)
y_train_pred = lm_final.predict(X_train_res)


# In[51]:


res = y_train - y_train_pred
sns.distplot(res)
plt.title('Error terms')
plt.show()


# In[52]:


c = [i for i in range(1,len(y_train)+1,1)]
fig = plt.figure(figsize=(8,5))
plt.scatter(y_train,res)
fig.suptitle('Error Terms', fontsize=16)
plt.xlabel('Y_train_pred', fontsize=14)
plt.ylabel('Residual', fontsize=14)  


# In[53]:


df_test.head()


# In[54]:


X_test_sm = X_test[var_final]
X_test_sm.head()


# In[57]:


X_test_sm = sm.add_constant(X_test_sm)
X_test_sm.head()


# In[58]:


y_test_pred = lm_final.predict(X_test_sm)


# In[59]:


r2_test = r2_score(y_true = y_test, y_pred = y_test_pred)
print('R-Squared for Test dataset: ', round(r2_test,3))


# In[60]:


N= len(X_test)
p =len(var_final)
r2_test_adj = round((1-((1-r2_test)*(N-1)/(N-p-1))),3)
print('Adj. R-Squared for Test dataset: ', round(r2_test_adj,3))


# In[61]:


mse = mean_squared_error(y_test, y_test_pred)
print('Mean_Squared_Error :' ,round(mse,4))


# In[62]:


res_test = y_test - y_test_pred
plt.title('Error Terms', fontsize=16) 
sns.distplot(res_test)
plt.show()


# In[63]:


c = [i for i in range(1,len(y_test)+1,1)]
fig = plt.figure(figsize=(8,5))
plt.scatter(y_test,res_test)
fig.suptitle('Error Terms', fontsize=16)
plt.xlabel('Y_test_pred', fontsize=14)
plt.ylabel('Residual', fontsize=14)   


# In[64]:


fig = plt.figure()
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)
plt.xlabel('y_test', fontsize = 18)
plt.ylabel('y_test_pred', fontsize = 16) 


# In[65]:


print('R- Sqaured train: ', round(lm_final.rsquared,2), '  Adj. R-Squared train:', round(lm_final.rsquared_adj,3) )
print('R- Sqaured test : ', round(r2_test,2), '  Adj. R-Squared test :', round(r2_test_adj,3))
print('\033[1m{:10s}\033[0m'.format('\nCoefficent for the variables are:'))
print(round(lm_final.params,3))


# In[ ]:




