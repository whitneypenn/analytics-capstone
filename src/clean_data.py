import pandas as pd
import numpy as np

class DataPipeline:
    '''
    Takes a Spotify Data CSV and gets it ready to be fed into a model.
    '''

    def __init__(self, data_file):
        self.data_file = data_file
        self.no_zeros = self.drop_zero_popularity(self.data_file)
        self.features, self.target = self.split_target_and_features(self.no_zeros)

    def drop_zero_popularity(self, data_file):
        return data_file[data_file['popularity'] > 1]

    def split_target_and_features(self, no_zeros):
        ## Keys into BCEG or Not
        key_list = [0, 4, 7, 11]
        bceg = no_zeros['key'].isin(key_list)
        no_zeros['bceg'] = bceg*1

        #Time Signature into categorical data
        # time_sig = pd.get_dummies(no_zeros['time_signature'])
        # time_sig = time_sig.rename(index = int, columns={1: "time_sig_1", 3: "time_sig_3", 4: "time_sig_4", 5:"time_sig_5"})
        # no_zeros = pd.concat([no_zeros, time_sig], axis=1)

        #features_dataframe_done = no_zeros.drop(['key', 'time_signature','mode'], axis=1)

        features = no_zeros[['acousticness', 'danceability',
           'duration_ms', 'energy','instrumentalness', 'liveness',
           'loudness', 'speechiness', 'tempo',
           'valence','bceg']]
        target = no_zeros['popularity']

        return features, target
