import numpy as np
import pandas as pd
import scipy.stats as st
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEBUG = True

# 1. Load and merge data

housing_df = pd.read_csv('data/train.csv')
test_df    = pd.read_csv('data/test.csv')
if DEBUG: print("load time shapes:", housing_df.shape, test_df.shape)

housing_df.set_index("Id", inplace=True)
test_df.set_index("Id", inplace=True)

# 2. Assign correct data types

for column in housing_df.select_dtypes(['object']).columns:
    housing_df[column] = housing_df[column].astype('category')
for column in test_df.select_dtypes(['object']).columns:
    test_df[column] = test_df[column].astype('category')

housing_df.MSSubClass = housing_df.MSSubClass.astype('category')
housing_df.OverallQual = housing_df.OverallQual.astype('category')
housing_df.OverallCond = housing_df.OverallCond.astype('category')
housing_df.BsmtFullBath = housing_df.BsmtFullBath.astype('category')
housing_df.BsmtHalfBath = housing_df.BsmtHalfBath.astype('category')
housing_df.FullBath = housing_df.FullBath.astype('category')
housing_df.HalfBath = housing_df.HalfBath.astype('category')
housing_df.BedroomAbvGr = housing_df.BedroomAbvGr.astype('category')
housing_df.KitchenAbvGr = housing_df.KitchenAbvGr.astype('category')
housing_df.TotRmsAbvGrd = housing_df.TotRmsAbvGrd.astype('category')
housing_df.Fireplaces = housing_df.Fireplaces.astype('category')
housing_df.GarageCars = housing_df.GarageCars.astype('category')
housing_df.MoSold = housing_df.MoSold.astype('category')

test_df.MSSubClass = test_df.MSSubClass.astype('category')
test_df.OverallQual = test_df.OverallQual.astype('category')
test_df.OverallCond = test_df.OverallCond.astype('category')
test_df.BsmtFullBath = test_df.BsmtFullBath.astype('category')
test_df.BsmtHalfBath = test_df.BsmtHalfBath.astype('category')
test_df.FullBath = test_df.FullBath.astype('category')
test_df.HalfBath = test_df.HalfBath.astype('category')
test_df.BedroomAbvGr = test_df.BedroomAbvGr.astype('category')
test_df.KitchenAbvGr = test_df.KitchenAbvGr.astype('category')
test_df.TotRmsAbvGrd = test_df.TotRmsAbvGrd.astype('category')
test_df.Fireplaces = test_df.Fireplaces.astype('category')
test_df.GarageCars = test_df.GarageCars.astype('category')
test_df.MoSold = test_df.MoSold.astype('category')

# 3. Handle missing values

housing_df.LotFrontage.fillna(housing_df.LotFrontage.mean(), inplace=True)
housing_df.MasVnrArea.fillna(housing_df.MasVnrArea.mean(), inplace=True)
housing_df.GarageYrBlt.fillna(housing_df.GarageYrBlt.mean(), inplace=True)

test_df.LotFrontage.fillna(housing_df.LotFrontage.mean(), inplace=True)
test_df.MasVnrArea.fillna(housing_df.MasVnrArea.mean(), inplace=True)
test_df.BsmtFinSF1.fillna(housing_df.BsmtFinSF1.mean(), inplace=True)
test_df.BsmtFinSF2.fillna(housing_df.BsmtFinSF2.mean(), inplace=True)
test_df.BsmtUnfSF.fillna(housing_df.BsmtUnfSF.mean(), inplace=True)
test_df.TotalBsmtSF.fillna(housing_df.TotalBsmtSF.mean(), inplace=True)
test_df.GarageYrBlt.fillna(housing_df.GarageYrBlt.mean(), inplace=True)
test_df.GarageArea.fillna(housing_df.GarageArea.mean(), inplace=True)

empty_means_without = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                        "BsmtFinType2", "FireplaceQu","GarageType","GarageFinish",
                        "GarageQual","GarageCond","PoolQC","Fence","MiscFeature",
                        'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                        'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional',
                        'GarageCars', 'SaleType']

def replace_empty(feature, value):
    housing_df[feature].cat.add_categories([value], inplace=True)
    test_df[feature].cat.add_categories([value], inplace=True)
    housing_df[feature].fillna(value, inplace=True)
    test_df[feature].fillna(value, inplace=True)

for feature in empty_means_without:
    replace_empty(feature, "None")

test_df.MasVnrType.cat.add_categories('without', inplace=True)
test_df.MasVnrType.fillna('without', inplace=True)
test_df.GarageCars.cat.add_categories('without', inplace=True)
test_df.GarageCars[test_df.GarageCars == 'None'] = 'without'

housing_df.dropna(inplace=True)

# 4. Preprocessing

numeric_df = housing_df.select_dtypes(exclude=['category']).copy()
numeric_test_df = test_df.select_dtypes(exclude=['category']).copy()

target = numeric_df['SalePrice']
numeric_df.drop("SalePrice", axis=1, inplace=True)

if DEBUG: print("numeric shapes:", numeric_df.shape, numeric_test_df.shape)

## one-hot encode
categorical_df = housing_df.select_dtypes(include=['category']).copy()
categorical_encoded_df = pd.get_dummies(categorical_df)
categorical_test_df = test_df.select_dtypes(include=['category']).copy()
categorical_test_encoded_df = pd.get_dummies(categorical_test_df)

categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('GarageCars_4.0' , 'GarageCars_4')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('BsmtFullBath_3.0', 'BsmtFullBath_3')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('GarageCars_2.0', 'GarageCars_2')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('BsmtHalfBath_2.0', 'BsmtHalfBath_2')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('GarageCars_3.0', 'GarageCars_3')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('GarageCars_0.0', 'GarageCars_0')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('GarageCars_5.0', 'GarageCars_5')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('BsmtHalfBath_0.0', 'BsmtHalfBath_0')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('BsmtHalfBath_1.0', 'BsmtHalfBath_1')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('GarageCars_1.0', 'GarageCars_1')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('BsmtFullBath_2.0' , 'BsmtFullBath_2')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('BsmtFullBath_1.0', 'BsmtFullBath_1')
categorical_test_encoded_df.columns = categorical_test_encoded_df.columns.str.replace('BsmtFullBath_0.0', 'BsmtFullBath_0')
categorical_test_encoded_df['Utilities_NoSeWa'] = 0
categorical_test_encoded_df['Electrical_Mix'] = 0
categorical_test_encoded_df['GarageQual_Ex'] = 0
categorical_test_encoded_df['Exterior2nd_Other'] = 0
categorical_test_encoded_df['RoofMatl_Roll'] = 0
categorical_test_encoded_df['TotRmsAbvGrd_2'] = 0
categorical_test_encoded_df['RoofMatl_Metal'] = 0
categorical_test_encoded_df['Heating_OthW'] = 0
categorical_test_encoded_df['RoofMatl_Membran'] = 0
categorical_test_encoded_df['Heating_Floor'] = 0
categorical_test_encoded_df['Condition2_RRAe'] = 0
categorical_test_encoded_df['Exterior1st_Stone'] = 0
categorical_test_encoded_df['Condition2_RRAn'] = 0
categorical_test_encoded_df['KitchenAbvGr_3'] = 0
categorical_test_encoded_df['MiscFeature_TenC'] = 0
categorical_test_encoded_df['Exterior1st_ImStucc'] = 0
categorical_test_encoded_df['RoofMatl_ClyTile'] = 0
categorical_test_encoded_df['BedroomAbvGr_8'] = 0
categorical_test_encoded_df['TotRmsAbvGrd_14'] = 0
categorical_test_encoded_df['HouseStyle_2.5Fin'] = 0
categorical_test_encoded_df['PoolQC_Fa'] = 0
categorical_test_encoded_df['Condition2_RRNn'] = 0

categorical_encoded_df['GarageCars_5'] = 0
categorical_encoded_df['FullBath_4'] = 0
categorical_encoded_df['GarageCars_without'] = 0
categorical_encoded_df['Fireplaces_4'] = 0
categorical_encoded_df['TotRmsAbvGrd_15'] = 0
categorical_encoded_df['TotRmsAbvGrd_13'] = 0
categorical_encoded_df['MasVnrType_without'] = 0
categorical_encoded_df['MSSubClass_150'] = 0

categorical_test_encoded_df = categorical_test_encoded_df[categorical_encoded_df.columns]

if DEBUG: print("categorical shapes:", categorical_df.shape, categorical_test_df.shape)
if DEBUG: print("categorical encoded shapes:", categorical_encoded_df.shape, categorical_test_encoded_df.shape)
if DEBUG: print(set(categorical_encoded_df.columns) - set(categorical_test_encoded_df.columns))
if DEBUG: print(set(categorical_test_encoded_df.columns) - set(categorical_encoded_df.columns))

## log transform
numeric_log_df = np.log(numeric_df + 1)
numeric_test_log_df = np.log(numeric_test_df + 1)

## scale the data
scaler = StandardScaler()
scaler.fit(numeric_log_df)
numeric_log_std_sc = scaler.transform(numeric_log_df)
numeric_test_log_std_sc = scaler.transform(numeric_test_log_df)

numeric_log_std_sc_df = pd.DataFrame(numeric_log_std_sc,
                                     columns=numeric_log_df.columns,
                                     index=numeric_log_df.index)
numeric_test_log_std_sc_df = pd.DataFrame(numeric_test_log_std_sc,
                                     columns=numeric_test_log_df.columns,
                                     index=numeric_test_log_df.index)

# outliers

def feature_outliers(dataframe, col, param=1.5):
    Q1 = np.percentile(dataframe[col], 25)
    Q3 = np.percentile(dataframe[col], 75)
    tukey_window = param*(Q3-Q1)
    less_than_Q1 = dataframe[col] < Q1 - tukey_window
    greater_than_Q3 = dataframe[col] > Q3 + tukey_window
    tukey_mask = (less_than_Q1 | greater_than_Q3)
    return dataframe[tukey_mask]

def multiple_outliers(dataframe, count=2):
    raw_outliers = []
    for col in dataframe:
        outlier_df = feature_outliers(dataframe, col)
        raw_outliers += list(outlier_df.index)

    outlier_count = Counter(raw_outliers)
    outliers = [k for k,v in outlier_count.items() if v >= count]
    return outliers

numeric_log_std_sc_out_rem_df = numeric_log_std_sc_df.drop(multiple_outliers(numeric_log_std_sc_df, 5))
categorical_encoded_out_rem_df = categorical_encoded_df.drop(multiple_outliers(numeric_log_std_sc_df, 5))
target_log_std_sc_out_rem_df = target.drop(multiple_outliers(numeric_log_std_sc_df, 5))

# principal component analysis

pca_log_std_sc_out_rem = PCA(8)

pca_log_std_sc_out_rem.fit(numeric_log_std_sc_out_rem_df)
numeric_pca = \
    pca_log_std_sc_out_rem.transform(numeric_log_std_sc_out_rem_df)
numeric_test_pca = \
    pca_log_std_sc_out_rem.transform(numeric_test_log_std_sc_df)


numeric_log_std_sc_out_rem_pca_df = \
    pd.DataFrame(numeric_pca,
                 columns=['PC 1', 'PC 2', 'PC 3', 'PC 4',
                          'PC 5', 'PC 6', 'PC 7', 'PC 8'],
                 index=numeric_log_std_sc_out_rem_df.index)
numeric_test_log_std_sc_pca_df = \
    pd.DataFrame(numeric_test_pca,
                 columns=['PC 1', 'PC 2', 'PC 3', 'PC 4',
                          'PC 5', 'PC 6', 'PC 7', 'PC 8'],
                 index=numeric_test_log_std_sc_df.index)

dataset_1 = pd.merge(categorical_encoded_out_rem_df,
                     numeric_log_std_sc_out_rem_df,
                     left_index=True, right_index=True)
dataset_2 = pd.merge(dataset_1, numeric_log_std_sc_out_rem_pca_df,
                     left_index=True, right_index=True)

testset_1 = pd.merge(categorical_test_encoded_df,
                      numeric_test_log_std_sc_df,
                      left_index=True, right_index=True)
testset_2 = pd.merge(testset_1, numeric_test_log_std_sc_pca_df,
                     left_index=True, right_index=True)

target_1 = target_log_std_sc_out_rem_df
target_2 = target_log_std_sc_out_rem_df

assert dataset_1.isnull().sum().sum() == 0
assert dataset_2.isnull().sum().sum() == 0
assert testset_1.isnull().sum().sum() == 0
assert testset_2.isnull().sum().sum() == 0

del Counter
del column
del empty_means_without
del feature
del feature_outliers
del housing_df
del multiple_outliers
del replace_empty
del st
