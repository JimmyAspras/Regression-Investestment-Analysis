#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Please  construct  a  new  dataset  by  either  adding  two  independent  variables  or  removing  
#two independent  variables  from  finalsample.dta  dataset.  If  you  choose  to  add  two  independent variables,
#you could add any two independent variables that you think help explain stock returns. If  you  choose  to 
#remove  two  independent  variables,  you  could  remove  any  two  independent variables that already exist 
#in the finalsample.dta dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from pandas_datareader import data
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
plt.rcParams['figure.figsize'] = [20, 15]


# In[2]:


#atq=total assets
#ceqq=common/ordinary equity
#cheq=cash and short term investments
#dlttq=total long term debt
#epspiq=eps including extraordinary items
#saleq=sales/turnover
#dvpspq=dividends per share
#
regdata=pd.read_stata('/Users/jimmyaspras/Downloads/finalsample.dta')
regdata.columns


# In[3]:


regdata.dtypes


# In[4]:


regdata.sort_values(by=['datadate'], inplace=True)
regdata1=regdata[regdata['lagPrice2']>=5]#remove penny stocks
regdata1['Year']=regdata1['datadate'].dt.year
regdata1['Month']=regdata1['datadate'].dt.month
#set gvkey and datadate as the index
regdata1=regdata1.set_index(['gvkey','datadate'])
regdata1.head()


# In[5]:


#Split  your  new  dataset  into  training  and  testing  samples.  Testing  sample  should  include  data 
#with year>=2016. 
#
#Drop dvpspq and atq from the train/test data
#
train=regdata1[regdata1['Year']<2016]
X_train=train[['sic_2', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2']]


# In[6]:


#Set return as the dependent variable
Y_train=train[['ret']]


# In[7]:


test=regdata1[regdata1['Year']>=2016]
X_test=test[['sic_2', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2']]


# In[8]:


Y_test=test[['ret']]


# In[9]:


#Calculate avg monthly risk free return
rf1=pd.read_excel("/Users/jimmyaspras/Downloads/Treasury bill.xlsx")
rf1['rf']=rf1['DGS3MO']/1200
rf2=rf1[['Date','rf']].dropna()
rf2['Year']=rf2['Date'].dt.year
rf2['Month']=rf2['Date'].dt.month
rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


# In[10]:


#Import benchmark index return
indexret1=pd.read_stata("/Users/jimmyaspras/Downloads/Index return.dta")


# In[11]:


#Use LassoCV to run lasso regression and use the timeseriessplit for the cross validation.  Run lasso  regression 
#using  your  new  training  sample  and  report  the  selected  value  of  Alpha  and coefficients on all 
#the selected independent variables. 


# In[12]:


variable_name=X_test.columns.tolist() #get the independent variable names
tsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)
Lasso_m = LassoCV(cv=tsplit) #define the model
Lasso_m.fit(X_train,Y_train)#train the model
print("Alpha: ",Lasso_m.alpha_)


# In[13]:


coefficients_Lasso=pd.DataFrame(Lasso_m.coef_, columns=['coef'])
coefficients_Lasso.index=variable_name
print (coefficients_Lasso)


# In[14]:


#Predict returns vs market
lassocoef_select=coefficients_Lasso.query("coef!=0")
Y_predictlasso=pd.DataFrame(Lasso_m.predict(X_test), columns=['Y_predict'])
Y_testlasso=pd.DataFrame(Y_test).reset_index()
lassoComb1=pd.merge(Y_testlasso, Y_predictlasso, left_index=True,right_index=True,how='inner')
lassoComb1['Year']=lassoComb1['datadate'].dt.year
lassoComb1['Month']=lassoComb1['datadate'].dt.month
lassorank1=lassoComb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
lassorank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
lassostock_long1=pd.merge(lassoComb1,lassorank1,left_index=True, right_index=True)
lassostock_long2=lassostock_long1[lassostock_long1['Y_predict_rank']<=100]
lassostock_long2['datadate'].value_counts()
lassostock_long3=lassostock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
lassostock_long4=pd.merge(lassostock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
lassostock_long5=pd.merge(lassostock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
lassostock_long5['ret_rf']=lassostock_long5['ret']-lassostock_long5['rf']
lassostock_long5['ret_sp500']=lassostock_long5['ret']-lassostock_long5['sp500_ret_m']
lassostock_long5=sm.add_constant(lassostock_long5)
lassostock_long5.head()


# In[15]:


sm.OLS(lassostock_long5[['ret']],lassostock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[16]:


#Return of 1.06% above the market, statistically significant with p-value of 0.036


# In[17]:


#Sharpe ratio
lassoRet_rf=lassostock_long5[['ret_rf']]
lassoSR=(lassoRet_rf.mean()/lassoRet_rf.std())*np.sqrt(12)
lassoSR


# In[18]:


#Use RidgeCV to run ridge regression and use the timeseriessplit for the cross validation. Please create your own
#candidate values for the search of Alpha. Run ridge regression using your new training  sample  and  report  
#the  selected  value  of  Alpha  and  coefficients  on  all  the  independent variables.
ridgetsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)
alpha_candidate=np.linspace(0.001,10,20)
Ridge_m = RidgeCV(alphas=alpha_candidate, cv=ridgetsplit)
Ridge_m.fit(X_train,Y_train)#train the model
Ridge_m.alpha_


# In[19]:


#Alpha 0.001


# In[20]:


coefficients_Ridge=pd.DataFrame(Ridge_m.coef_).T
coefficients_Ridge.index=variable_name
print (coefficients_Ridge)


# In[21]:


Y_predictridge=pd.DataFrame(Ridge_m.predict(X_test), columns=['Y_predict']) 
Y_testridge=pd.DataFrame(Y_test).reset_index()
ridgeComb1=pd.merge(Y_testridge, Y_predictridge, left_index=True,right_index=True,how='inner')
ridgeComb1['Year']=ridgeComb1['datadate'].dt.year
ridgeComb1['Month']=ridgeComb1['datadate'].dt.month
ridgerank1=ridgeComb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
ridgerank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
ridgestock_long1=pd.merge(ridgeComb1,ridgerank1,left_index=True, right_index=True)
ridgestock_long2=ridgestock_long1[ridgestock_long1['Y_predict_rank']<=100]
ridgestock_long2['datadate'].value_counts()
ridgestock_long3=ridgestock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
ridgestock_long4=pd.merge(ridgestock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
ridgestock_long5=pd.merge(ridgestock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
ridgestock_long5['ret_rf']=ridgestock_long5['ret']-ridgestock_long5['rf']
ridgestock_long5['ret_sp500']=ridgestock_long5['ret']-ridgestock_long5['sp500_ret_m']
ridgestock_long5=sm.add_constant(ridgestock_long5)
ridgestock_long5.head()


# In[22]:


sm.OLS(ridgestock_long5[['ret']],ridgestock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[23]:


#The ridge regression returns a value of 36.99% average return per month with a p-value of 0


# In[24]:


ridgeRet_rf=ridgestock_long5[['ret_rf']]
ridgeSR=(ridgeRet_rf.mean()/ridgeRet_rf.std())*np.sqrt(12)
ridgeSR


# In[25]:


#Use ElasticNetCV to run elasticnet regression and use the timeseriessplit for the cross validation. Please 
#create your own candidate values for the search of l1_ratio. Run elasticnet regression using your  new  training 
#sample  and  report  the  selected  value  of  l1_ratio  and  coefficients  on  all  the selected independent 
#variables.
elasttsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)
l1_ratio_candidate=np.linspace(0.001,2,10)
Elastic_m = ElasticNetCV(l1_ratio=l1_ratio_candidate, cv=tsplit)
Elastic_m.fit(X_train,Y_train)
print("Chosen alpha value:",Elastic_m.l1_ratio_)


# In[26]:


coefficients_Elas=pd.DataFrame(Elastic_m.coef_,columns=['coef'])
coefficients_Elas.index=variable_name
print (coefficients_Elas)


# In[27]:


coef_select_Elas=coefficients_Lasso.query("coef!=0")
Y_predictelast=pd.DataFrame(Elastic_m.predict(X_test), columns=['Y_predict'])
Y_testelast=pd.DataFrame(Y_test).reset_index()
elastComb1=pd.merge(Y_testelast, Y_predictelast, left_index=True,right_index=True,how='inner')
elastComb1['Year']=elastComb1['datadate'].dt.year
elastComb1['Month']=elastComb1['datadate'].dt.month
elastrank1=elastComb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
elastrank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
elaststock_long1=pd.merge(elastComb1,elastrank1,left_index=True, right_index=True)
elaststock_long2=elaststock_long1[elaststock_long1['Y_predict_rank']<=100]
elaststock_long2['datadate'].value_counts()
elaststock_long3=elaststock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
elaststock_long4=pd.merge(elaststock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
elaststock_long5=pd.merge(elaststock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
elaststock_long5['ret_rf']=elaststock_long5['ret']-elaststock_long5['rf']
elaststock_long5['ret_sp500']=elaststock_long5['ret']-elaststock_long5['sp500_ret_m']
elaststock_long5=sm.add_constant(elaststock_long5)
elaststock_long5.head()


# In[28]:


sm.OLS(elaststock_long5[['ret']],elaststock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[29]:


elastRet_rf=elaststock_long5[['ret_rf']]
elastSR=(elastRet_rf.mean()/elastRet_rf.std())*np.sqrt(12)
elastSR


# In[30]:


#Results of ElasticNetCV are identical to lasso regression with a monthly return of 1.06 and, p-value of 0.036
#and sharpe ratio of 0.841747

