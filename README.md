# Regression-Investment-Analysis
Regression analysis of specific variables to beat the market

## Introduction

Can regression models be used to beat the market?

Data for this project was chosen and downloaded from Wharton Research Data Services: https://wrds-www.wharton.upenn.edu/. This was done as part of a course taken in Summer 2021 complete with prompts and analysis. Credit must be given to my professor, Dr. Wei Jiao, for much code and instruction included here.

This project follows a series of prompts to determine if a regression model can be refined to beat market return.

## Preparing the Data

**Construct a new dataset by either adding two independent variables or removing two independent variables from finalsample.dta dataset. If you choose to add two independent variables, you could add any two independent variables that you think help explain stock returns. If you choose to remove two independent  variables, you could remove any two independent variables that already exist in the finalsample.dta dataset.**

### Libraries

```python
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
```

### Importing the Data

```python
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
```
Output:
```python
Index(['gvkey', 'datadate', 'sic_2', 'lagdate', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'lagdatadate', 'atq', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'dvpspq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2'],
      dtype='object')
```

We want to sort the data by date and exclude penny stocks. We also create vectors for year and month for the analysis and set the stock key and datadate as the index.

```python
regdata.sort_values(by=['datadate'], inplace=True)
regdata1=regdata[regdata['lagPrice2']>=5]#remove penny stocks
regdata1['Year']=regdata1['datadate'].dt.year
regdata1['Month']=regdata1['datadate'].dt.month
#set gvkey and datadate as the index
regdata1=regdata1.set_index(['gvkey','datadate'])
regdata1.head()
```

**Split your new dataset into training and testing samples. Testing sample should include data with year>=2016. I chose to drop dvpspq (dividends) and atq (total assets) from the train/test data**

```python
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
```

Set return as the dependent variable
```python
Y_train=train[['ret']]
```

```
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
```

```python
Y_test=test[['ret']]
```

Calculate avg monthly risk free return
```python
rf1=pd.read_excel("/Users/jimmyaspras/Downloads/Treasury bill.xlsx")
rf1['rf']=rf1['DGS3MO']/1200
rf2=rf1[['Date','rf']].dropna()
rf2['Year']=rf2['Date'].dt.year
rf2['Month']=rf2['Date'].dt.month
rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()
```

Import benchmark index return
```python
indexret1=pd.read_stata("/Users/jimmyaspras/Downloads/Index return.dta")
```

### Building the First Model - Lasso Regression

**Use LassoCV to run lasso regression and use the timeseriessplit for the cross validation. Run lasso regression using your new training sample and report  the selected value of Alpha and coefficients on all the selected independent variables.**

```python
variable_name=X_test.columns.tolist() #get the independent variable names
tsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)
Lasso_m = LassoCV(cv=tsplit) #define the model
Lasso_m.fit(X_train,Y_train)#train the model
print("Alpha: ",Lasso_m.alpha_)
```

Output:
```python
Alpha:  87.13472491755509
```

Display the coefficients

```python
coefficients_Lasso=pd.DataFrame(Lasso_m.coef_, columns=['coef'])
coefficients_Lasso.index=variable_name
print (coefficients_Lasso)
```

```python
                            coef
sic_2              -0.000000e+00
lagRet2             0.000000e+00
lagVOL2            -8.656721e-12
lagPrice2          -0.000000e+00
lagMV2             -0.000000e+00
lagShareturnover2  -0.000000e+00
lagRet2_sic         0.000000e+00
lagRet12            0.000000e+00
lagVOL12           -8.381372e-13
lagShareturnover12 -0.000000e+00
lagRet12_std       -0.000000e+00
lagRet12_min        0.000000e+00
lagRet12_max       -0.000000e+00
lagRet12_sic        0.000000e+00
ceqq               -0.000000e+00
cheq               -0.000000e+00
dlttq              -0.000000e+00
epspiq             -0.000000e+00
saleq              -0.000000e+00
sp500_ret_d         0.000000e+00
nasdaq_ret_d        0.000000e+00
r2000_ret_d         0.000000e+00
dollar_ret_d        0.000000e+00
VIX                -0.000000e+00
yield_3m           -0.000000e+00
yield_10y          -0.000000e+00
gdp_growth         -0.000000e+00
Bull_ave            0.000000e+00
Bull_Bear           0.000000e+00
ret                 0.000000e+00
debt                0.000000e+00
cash               -0.000000e+00
sale                0.000000e+00
BM                  0.000000e+00
PE                 -0.000000e+00
div_p              -0.000000e+00
loglagPrice2       -0.000000e+00
loglagVOL12        -0.000000e+00
loglagMV2          -0.000000e+00
logatq             -0.000000e+00
loglagVOL2         -0.000000e+00
```

Predict returns vs market
```python
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
 ```
 
 Model Summary
 ```python
 sm.OLS(lassostock_long5[['ret']],lassostock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
```

<img width="393" alt="image" src="https://user-images.githubusercontent.com/72087263/188288717-50c296b5-31ed-4c21-90ad-48c8364b66a9.png">

**The Lasso Regression model has a return of 1.06% above the market and is statistically significant with p-value of 0.036.**

Sharpe ratio
```python
lassoRet_rf=lassostock_long5[['ret_rf']]
lassoSR=(lassoRet_rf.mean()/lassoRet_rf.std())*np.sqrt(12)
lassoSR
```

```python
ret_rf    0.841747
```

### Building the Second Model - RidgeCV

**Use RidgeCV to run ridge regression and use the timeseriessplit for the cross validation. Please create your own candidate values for the search of Alpha. Run ridge regression using your new training sample and report the selected value of Alpha and coefficients on all the independent variables.
```python
ridgetsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)
alpha_candidate=np.linspace(0.001,10,20)
Ridge_m = RidgeCV(alphas=alpha_candidate, cv=ridgetsplit)
Ridge_m.fit(X_train,Y_train)#train the model
Ridge_m.alpha_
```

```python
Alpha: 0.001
```

Display coefficients
```python
coefficients_Ridge=pd.DataFrame(Ridge_m.coef_).T
coefficients_Ridge.index=variable_name
print (coefficients_Ridge)
```

```python
                               0
sic_2              -3.387688e-12
lagRet2             4.065479e-11
lagVOL2             4.222271e-19
lagPrice2           4.382608e-14
lagMV2              2.438675e-15
lagShareturnover2  -2.702210e-09
lagRet2_sic         1.676743e-10
lagRet12            3.604271e-08
lagVOL12           -3.491729e-19
lagShareturnover12  3.053015e-09
lagRet12_std       -2.229274e-08
lagRet12_min       -6.062082e-09
lagRet12_max        3.397773e-09
lagRet12_sic       -1.576521e-10
ceqq               -7.032788e-15
cheq                4.233701e-15
dlttq              -9.201331e-15
epspiq             -2.367641e-13
saleq              -4.395994e-15
sp500_ret_d         1.526742e-06
nasdaq_ret_d       -1.800111e-07
r2000_ret_d        -7.259178e-07
dollar_ret_d        3.271167e-07
VIX                 1.027323e-10
yield_3m            5.465529e-07
yield_10y          -9.181538e-07
gdp_growth         -1.334378e-08
Bull_ave            6.090713e-09
Bull_Bear           4.048946e-09
ret                 9.999999e-01
debt                6.386898e-10
cash                9.225993e-10
sale                1.492856e-09
BM                  9.499182e-10
PE                 -1.175696e-13
div_p              -3.470031e-09
loglagPrice2       -6.081579e-10
loglagVOL12         6.275039e-11
loglagMV2           3.383554e-11
logatq              1.548527e-10
loglagVOL2         -1.729408e-10
```

Calculate model return vs the market
```python
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
```

Model Summary
```python
sm.OLS(ridgestock_long5[['ret']],ridgestock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
```

<img width="397" alt="image" src="https://user-images.githubusercontent.com/72087263/188289406-9af8f0dd-e86c-4994-989a-c3d19f54fa87.png">

**The ridge regression returns a value of 36.99% average return per month with a p-value of 0.**

Sharpe Ratio
```python
ridgeRet_rf=ridgestock_long5[['ret_rf']]
ridgeSR=(ridgeRet_rf.mean()/ridgeRet_rf.std())*np.sqrt(12)
ridgeSR
```
```python
ret_rf    8.458599
```

### Builing the Third Model - Elastic Net

**Use ElasticNetCV to run elasticnet regression and use the timeseriessplit for the cross validation. Please create your own candidate values for the search of l1_ratio. Run elasticnet regression using your new training sample and report the selected value of l1_ratio and coefficients on all the selected independent variables.**

```python
elasttsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)
l1_ratio_candidate=np.linspace(0.001,2,10)
Elastic_m = ElasticNetCV(l1_ratio=l1_ratio_candidate, cv=tsplit)
Elastic_m.fit(X_train,Y_train)
print("Chosen alpha value:",Elastic_m.l1_ratio_)
```

```python
Chosen alpha value: 0.001
```

Display Coefficients
```python
coefficients_Elas=pd.DataFrame(Elastic_m.coef_,columns=['coef'])
coefficients_Elas.index=variable_name
print (coefficients_Elas)
```

```python
                            coef
sic_2              -0.000000e+00
lagRet2             0.000000e+00
lagVOL2            -8.656721e-12
lagPrice2          -0.000000e+00
lagMV2             -0.000000e+00
lagShareturnover2  -0.000000e+00
lagRet2_sic         0.000000e+00
lagRet12            0.000000e+00
lagVOL12           -8.381372e-13
lagShareturnover12 -0.000000e+00
lagRet12_std       -0.000000e+00
lagRet12_min        0.000000e+00
lagRet12_max       -0.000000e+00
lagRet12_sic        0.000000e+00
ceqq               -0.000000e+00
cheq               -0.000000e+00
dlttq              -0.000000e+00
epspiq             -0.000000e+00
saleq              -0.000000e+00
sp500_ret_d         0.000000e+00
nasdaq_ret_d        0.000000e+00
r2000_ret_d         0.000000e+00
dollar_ret_d        0.000000e+00
VIX                -0.000000e+00
yield_3m           -0.000000e+00
yield_10y          -0.000000e+00
gdp_growth         -0.000000e+00
Bull_ave            0.000000e+00
Bull_Bear           0.000000e+00
ret                 0.000000e+00
debt                0.000000e+00
cash               -0.000000e+00
sale                0.000000e+00
BM                  0.000000e+00
PE                 -0.000000e+00
div_p              -0.000000e+00
loglagPrice2       -0.000000e+00
loglagVOL12        -0.000000e+00
loglagMV2          -0.000000e+00
logatq             -0.000000e+00
loglagVOL2         -0.000000e+00
```

Calculate returns vs the market
```python
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
```

Model Summary
```python
sm.OLS(elaststock_long5[['ret']],elaststock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
```

<img width="394" alt="image" src="https://user-images.githubusercontent.com/72087263/188289648-4b2397b0-32cf-46fc-b82d-4b65fa43ed8b.png">

Sharpe Ratio
```python
elastRet_rf=elaststock_long5[['ret_rf']]
elastSR=(elastRet_rf.mean()/elastRet_rf.std())*np.sqrt(12)
elastSR
```

```python
ret_rf    0.841747
```

**Results of ElasticNetCV are identical to lasso regression with a monthly return of 1.06 and, p-value of 0.036 and sharpe ratio of 0.841747**

## Conclusion

The Ridge Regression model performs the best out of the three tested regression models with an excess return of 37% and a Sharpe Ratio of 8.46. The Lasso and ElasticNetCV Regression models, while statistically significant, each only produced an excess return of 1.06% and a Sharpe Ratio of 0.84. Therefore, these regression models can be used to beat the market.
