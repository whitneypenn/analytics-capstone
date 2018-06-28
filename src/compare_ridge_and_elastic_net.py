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

def compare_elastic_models_over_alphas(alpha_range, X_train_std, y_train):
    kf = KFold(10)
    scorers = ['r2', 'neg_mean_squared_error']
    best_test_score_per_alpha = []
    best_train_score_per_alpha = []
    best_train_rmse_per_alpha = []
    best_test_rmse_per_alpha = []
    for alpha in alpha_range:
        elastic_model = ElasticNet(alpha=alpha, l1_ratio=.25)
        elastic_scores = cross_validate(elastic_model, X_train_std, y_train, scoring=scorers, cv=kf, return_train_score=True)
        best_test_score_per_alpha.append(elastic_scores['test_r2'].max())
        best_train_score_per_alpha.append(elastic_scores['train_r2'].max())
        best_test_rmse_per_alpha.append(np.sqrt(np.abs(elastic_scores['test_neg_mean_squared_error'].max())))
        best_train_rmse_per_alpha.append(np.sqrt(np.abs(elastic_scores['train_neg_mean_squared_error'].max())))
    return (np.array(best_test_score_per_alpha),
           np.array(best_train_score_per_alpha),
           np.array(best_test_rmse_per_alpha),
           np.array(best_train_rmse_per_alpha))


def compare_ridge_over_alphas(alpha_range, X_train_std, y_train):
    kf = KFold(10)
    scorers = ['r2', 'neg_mean_squared_error']
    best_test_score_per_alpha = []
    best_train_score_per_alpha = []
    best_train_rmse_per_alpha = []
    best_test_rmse_per_alpha = []
    for alpha in alpha_range:
            ridge_model = Ridge(alpha=alpha)
            ridge_scores = cross_validate(ridge_model, X_train_std, y_train, scoring=scorers, cv=kf, return_train_score=True)
            best_test_score_per_alpha.append(ridge_scores['test_r2'].max())
            best_train_score_per_alpha.append(ridge_scores['train_r2'].max())
            best_test_rmse_per_alpha.append(np.sqrt(np.abs(ridge_scores['test_neg_mean_squared_error'].max())))
            best_train_rmse_per_alpha.append(np.sqrt(np.abs(ridge_scores['train_neg_mean_squared_error'].max())))
    return (np.array(best_test_score_per_alpha),
            np.array(best_train_score_per_alpha),
            np.array(best_test_rmse_per_alpha),
            np.array(best_train_rmse_per_alpha))


def plot_R2_alpha_comparisons(alpha_range, test_r2, train_r2, model_name):
    sns.set(font_scale=2)
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)
    ax.set_title('$R^2$ For '+ model_name)
    ax.set_xlabel('Alphas')
    ax.set_ylabel('$R^2$')
    plt.plot(alpha_range, test_r2, label="test r^2")
    plt.plot(alpha_range, train_r2, label="train r^2")
    plt.legend()
    #plt.show()
    filename = 'images/'+model_name+'plot_R2_alpha_comparisons.png'
    plt.savefig(filename)
    plt.close()

def plot_RMSE_alpha_comparisons(alpha_range, test_rmse, train_rmse, model_name):
    sns.set(font_scale=2)
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)
    ax.set_title('RMSE For ' + model_name)
    ax.set_xlabel('Alphas')
    ax.set_ylabel('RMSE')
    plt.plot(alpha_range, test_rmse, label = 'test rmse')
    plt.plot(alpha_range, train_rmse, label = 'train rmse')
    plt.legend()
    #plt.show()
    filename = 'images/'+model_name+'plot_RMSE_alpha_comparisons.png'
    plt.savefig(filename)
    plt.close()

def print_stats(model_name, test_r2, train_r2, test_rmse, train_rmse):
    print(model_name)
    print('Test R2 Train R2')
    print(test_r2.max(), train_r2.max())
    print("Test RMSE Train RMSE")
    print(test_rmse.max(), train_rmse.max())

data = pd.read_csv('~/Galvanize/analytics-capstone/data/spotify_data.csv')
all_data = DataPipeline(data)
X = all_data.features
y = all_data.target

elastic_a_range = np.linspace(.001, 50, 1000)
ridge_a_range = np.linspace(.01, 1000, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

elastic_test_r2, elastic_train_r2, elastic_test_rmse, elastic_train_rmse = compare_elastic_models_over_alphas(elastic_a_range, X_train, y_train)
print_stats("Elastic", elastic_test_r2, elastic_train_r2, elastic_test_rmse, elastic_train_rmse )
plot_R2_alpha_comparisons(elastic_a_range, elastic_test_r2, elastic_train_r2, 'Elastic Net Model with L1 Ratio .25')
plot_RMSE_alpha_comparisons(elastic_a_range, elastic_test_rmse, elastic_train_rmse, 'Elastic Net Model with L1 Ratio .25')

ridge_test_r2, ridge_train_r2, ridge_test_rmse, ridge_train_rmse = compare_ridge_over_alphas(ridge_a_range, X_train, y_train)
print_stats("Ridge", ridge_test_r2, ridge_train_r2, ridge_test_rmse, ridge_train_rmse)
plot_R2_alpha_comparisons(ridge_a_range, ridge_test_r2, ridge_train_r2, 'Ridge Model')
plot_RMSE_alpha_comparisons(ridge_a_range, ridge_test_rmse, ridge_train_rmse, 'Ridge Model')
