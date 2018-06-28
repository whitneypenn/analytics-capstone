import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set(color_codes=True)
from clean_data import DataPipeline
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def popularity_dist_plot(targets):
    sns.set(font_scale=2)
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)
    plot = sns.distplot(targets, kde=False, rug=False, color='green')
    plt.savefig('images/popularity_hist.png')
    plt.close()

def feature_distribution_plots(features):
    sns.set(font_scale=2)
    for i in features:
        fig, ax = plt.subplots()
        fig.set_size_inches(11, 8)
        plt.xlabel(i)
        plt.ylabel('count')
        sns.distplot(features[i], kde=False, color='green')
        filename = 'images/'+ i + '_dist_plot_plt.png'
        plt.savefig(filename)
        plt.close()

def target_vs_feature_scatter(targets, features):
    for i in features:
        fig, ax = plt.subplots()
        fig.set_size_inches(11, 8)
        sns.regplot(features[i], targets, fit_reg=False, color="green")
        plt.xlabel(i)
        plt.ylabel('popularity')
        filename = 'images/'+ i + '_scatter_plt.png'
        plt.savefig(filename)
        plt.close()

def find_colinearity(features):
    features = add_constant(features)
    return pd.Series([variance_inflation_factor(features.values, i)
               for i in range(features.shape[1])],
              index=features.columns)


data = pd.read_csv('~/Galvanize/analytics-capstone/data/spotify_data_2.csv')
X = DataPipeline(data)
popularity_dist_plot(X.target)
feature_distribution_plots(X.features)
target_vs_feature_scatter(X.target, X.features)
colinearity = find_colinearity(X.features)
