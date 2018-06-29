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

class EDA:

    def __init__(self, targets, features):
        self.targets = targets
        self.features = features

    def target_dist_plot(self, save=False, file_name=None):
        sns.set(font_scale=2)
        fig, ax = plt.subplots()
        fig.set_size_inches(11, 8)
        plot = sns.distplot(self.targets, kde=False, rug=False, color='green')
        if save:
            filename = 'images/'+file_name+'.png'
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


    def feature_distribution_plots(self, save=False, file_name=None):
        sns.set(font_scale=2)
        for i in self.features:
            fig, ax = plt.subplots()
            fig.set_size_inches(11, 8)
            plt.xlabel(i)
            plt.ylabel('count')
            sns.distplot(self.features[i], kde=False, color='green')
            if save:
                filename = filename = 'images/'+ i + '_dist_plot_plt.png'
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()

    def target_vs_feature_scatter(self, save=False, file_name=None):
        for i in self.features:
            fig, ax = plt.subplots()
            fig.set_size_inches(11, 8)
            sns.regplot(self.features[i], self.targets, fit_reg=False, color="green")
            plt.xlabel(i)
            plt.ylabel('popularity')
            if save:
                filename = filename = 'images/'+ i + '_scatter_plt.png'
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()

    def find_colinearity(self):
        features_with_const = add_constant(self.features)
        return print(pd.Series([variance_inflation_factor(features_with_const.values, i)
                   for i in range(features_with_const.shape[1])],
                  index=features_with_const.columns))

if __name__ == "__main__":
    data = pd.read_csv('~/Galvanize/analytics-capstone/data/spotify_data.csv')
    X = DataPipeline(data)
    eda = EDA(X.target, X.features)
    eda.target_dist_plot()
    eda.feature_distribution_plots()
    eda.target_vs_feature_scatter()
    colin = eda.find_colinearity()
