import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline

adult_train_data = pd.read_pickle('data/adult_train_data.p')
adult_train_labels = pd.read_pickle('data/adult_train_labels.p')

adult_test_data = pd.read_pickle('data/adult_test_data.p')
adult_test_labels = pd.read_pickle('data/adult_test_labels.p')

import sys

data = {
    'adult' : {
        'train' : {
            'raw_data' : adult_train_data,
            'labels' : adult_train_labels
        },
        'test' : {
            'raw_data' : adult_test_data,
            'labels' : adult_test_labels
        }
    }
}

numeric_features = ['age','capital-gain','capital-loss','hours-per-week']

def adult_feature_engineering(train_data, test_data):

    train_data = train_data.drop('education-num', axis=1)
    test_data = test_data.drop('education-num', axis=1)

    train_data = train_data.drop('fnlwgt', axis=1)
    test_data = test_data.drop('fnlwgt', axis=1)

    train_data = pd.get_dummies(train_data)
    test_data = pd.get_dummies(test_data)

    test_data['native-country_ Holand-Netherlands'] = np.zeros_like(test_data.age)

    # ensure column order of test is exactly the same as train
    test_data = test_data.reindex(columns = train_data.columns)
    
    numeric_data = train_data[numeric_features].copy()
    numeric_data_scaled = (numeric_data - numeric_data.mean())/(2*numeric_data.std())
    train_data[numeric_features] = numeric_data

    return train_data, test_data


data['adult']['train']['engineered'], \
    data['adult']['test']['engineered'] = \
        adult_feature_engineering(data['adult']['train']['raw_data'].copy(),
                                  data['adult']['test']['raw_data'].copy())
