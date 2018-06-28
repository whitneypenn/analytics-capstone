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
import statsmodels.api as sm

#Import Training Data
training_data = pd.read_csv('~/Galvanize/analytics-capstone/data/spotify_data.csv')
training_data = DataPipeline(training_data)
train_X = training_data.features
train_X = sm.add_constant(train_X)
train_y = training_data.target

#Import Unseen Playlist Data
unseen_data = pd.read_csv('~/Galvanize/analytics-capstone/data/unseen_data.csv')
unseen_data = DataPipeline(unseen_data)
test_X = unseen_data.features
test_X = sm.add_constant(test_X)
test_y = unseen_data.target

#Standardize Data
scalar = StandardScaler()
X_train_std = scalar.fit_transform(train_X)
X_test_std = scalar.fit_transform(test_X)

#Train a Ridge Model
ridge_model = Ridge()
ridge_model.fit(X_train_std, train_y)
ridge_predictions = ridge_model.predict(X_test_std)
print(r2_score(test_y, ridge_predictions))
print(np.sqrt(mean_squared_error(test_y, ridge_predictions)))
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
sns.regplot(ridge_predictions, test_y, fit_reg=False, color="green")
ax.set_xlabel('Predicted Popularity')
ax.set_ylabel('Actual Popularity')
ax.set_title("Predicted vs. Actual Popularity for Ridge Model")
plt.savefig('images/predicted_vs_actual_ridge')
plt.show()

#Train an OLS Model
ols_model = LinearRegression()
ols_model.fit(X_train_std, train_y)
ols_predictions = ridge_model.predict(X_test_std)
print(r2_score(test_y, ols_predictions))
print(np.sqrt(mean_squared_error(test_y, ols_predictions)))
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
sns.regplot(ols_predictions, test_y, fit_reg=False, color="green")
ax.set_xlabel('Predicted Popularity')
ax.set_ylabel('Actual Popularity')
ax.set_title("Predicted vs. Actual Popularity for OLS Model")
plt.savefig('images/predicted_vs_actual_ols')
plt.show()
