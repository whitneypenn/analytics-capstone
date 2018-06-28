import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from clean_data import DataPipeline

data = pd.read_csv('~/Galvanize/analytics-capstone/data/spotify_data.csv')
all_data = DataPipeline(data)
X = all_data.features
y = all_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

kf = KFold(10)
scorers = ['r2', 'neg_mean_squared_error']
test_scores = ()
train_scores = ()

#normalize data
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train)
X_test_std = scalar.fit_transform(X_test)

# OLS
ols_model = LinearRegression()
ols_scores = cross_validate(ols_model, X_train, y_train, scoring=scorers, cv=kf, return_train_score=True)

#L1 / Lasso
lasso_model = Lasso()
lasso_scores = cross_validate(lasso_model, X_train, y_train, scoring=scorers, cv=kf, return_train_score=True)

#L2 / Ridge
ridge_model = Ridge()
ridge_scores = cross_validate(ridge_model, X_train, y_train, scoring=scorers, cv=kf, return_train_score=True)


#ElasticNet
elastic_model = ElasticNet()
elastic_scores = cross_validate(elastic_model, X_train, y_train, scoring=scorers, cv=kf, return_train_score=True)

list_of_scores = [ols_scores, lasso_scores, ridge_scores, elastic_scores]
list_of_model_names = ['OLS', 'Lasso', 'Ridge', 'ElasticNet']
j = 0
for i in list_of_scores:
    print(list_of_model_names[j])
    print('train r2', i['train_r2'].max())
    print('test r2: ', i['test_r2'].max())
    print('train rmse', np.sqrt(np.abs(i['train_neg_mean_squared_error'].max())))
    print('test rmse', np.sqrt(np.abs(i['test_neg_mean_squared_error'].max())))
    print(' ')
    j += 1
