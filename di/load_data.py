import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
import time
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import TensorDataset, DataLoader


""" 2 Gaussian toy dataset """

def load_toy_dataset(n = 10000, dim = 20) :
    
    xs_train_sub, y_train = make_blobs(n_samples = n, n_features = dim, centers = 2)
    xs_test_sub, y_test = make_blobs(n_samples = int(n/2), n_features = dim, centers = 2)
    
    x_train, x_test = xs_train_sub[:, :-1], xs_test_sub[:, :-1]
    s_train, s_test = xs_train_sub[:, -1], xs_test_sub[:, -1]
    
    # toy sensitives
    s_train, s_test = (s_train > 0) * 1, (s_test > 0) * 1
    
    xs_train = np.hstack((x_train, s_train.reshape(s_train.shape[0], 1)))
    xs_test = np.hstack((x_test, s_test.reshape(s_test.shape[0], 1)))
   
    return (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test)


""" ADULT dataset """

def load_adult_dataset(lfr = False, minmax = True, sensitive = "gender", seed = 2021, testsize = 0.40):
    # The continuous variable fnlwgt represents final weight, which is the
    # number of units in the target population that the responding unit
    # represents.
    df_train = pd.read_csv("datasets/adult/adult.data", header=None)
    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "salary"]
    df_train.columns = columns
    df_test = pd.read_csv("datasets/adult/adult.test", header=None, comment="|")
    df_test.columns = columns

    def proc_z(Z):
        return np.array([1 if "Male" in z else 0 for z in Z])
    
    def proc_r(R):
        return np.array([1 if "White" in r else 0 for r in R])

    def proc_y(Y):
        return np.array([1 if ">50K" in y else 0 for y in Y])

    
    Y_train, Y_test = [proc_y(s["salary"]) for s in [df_train, df_test]]

    col_quanti = ["age", "education-num", "capital-gain",
                  "capital-loss", "hours-per-week"]  # "fnlwgt",
    
    if sensitive == "gender" :
        col_quali = ["workclass", "marital-status", "occupation",
                     "relationship", "race", "native-country"]
        Z_train, Z_test = [proc_z(s["sex"]) for s in [df_train, df_test]]
        
    elif sensitive == "race" :
        col_quali = ["workclass", "marital-status", "occupation",
                     "relationship", "sex", "native-country"]
        Z_train, Z_test = [proc_r(s["race"]) for s in [df_train, df_test]]
        
        
    else :
        raise NotImplementedError
        
        
    X_train_quali = df_train[col_quali].values
    X_test_quali = df_test[col_quali].values

    X_train_quanti = df_train[col_quanti]
    X_test_quanti = df_test[col_quanti]

    quali_encoder = OneHotEncoder(categories="auto", drop="first")
    quali_encoder.fit(X_train_quali)

    X_train_quali_enc = quali_encoder.transform(X_train_quali).toarray()
    X_test_quali_enc = quali_encoder.transform(X_test_quali).toarray()
    
    # drop Transport-moving in occupation: occupation의 다른 dummy 변수들의 선형 결함으로 설명가능
    X_train_quali_enc = np.delete(X_train_quali_enc, 27, axis=1)#.shape
    X_test_quali_enc = np.delete(X_test_quali_enc, 27, axis=1)
    
    X_train = np.concatenate([X_train_quali_enc, X_train_quanti], axis=1)
    X_test = np.concatenate([X_test_quali_enc, X_test_quanti], axis=1)

    scaler = StandardScaler()
    minmaxscaler = MinMaxScaler(feature_range = (0.0, 1.0)) # we do minmax scaling
    
    X_train = scaler.fit_transform(X_train)
    if minmax : 
        X_train = minmaxscaler.fit_transform(X_train) # we do minmax scaling
    
    X_test = scaler.transform(X_test)
    if minmax : 
        X_test = minmaxscaler.transform(X_test) # we do minmax scaling
    
    if lfr :
        X_train = np.load("x_reps.npy")
        X_test = np.load("test_x_reps.npy")
        
    XZ_train = np.concatenate([X_train, Z_train.reshape(Z_train.shape[0], 1)], axis = 1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(Z_test.shape[0], 1)], axis = 1)
   
    ### all and split again
    XZ = np.concatenate([XZ_train, XZ_test])
    X = np.concatenate([X_train, X_test])
    Y = np.concatenate([Y_train, Y_test])
    Z = np.concatenate([Z_train, Z_test])
    
    XZ_train, XZ_test, X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(XZ, X, Y, Z, test_size=testsize, random_state=seed)   

    return (XZ_train, X_train, Y_train, Z_train), (XZ_test, X_test, Y_test, Z_test)



""" BANK dataset """

def load_bank_dataset(seed = 42, testsize = 0.40, minmax = True):
    # It is bank marketing data.
    # bank.csv 462K lines 450 Ko
    # bank-full 4M 614K lines 4.4 Mo
    # https://archive.ics.uci.edu/ml/datasets/bank+marketing
    
    df = pd.read_csv("datasets/bank/bank-additional-full.csv", sep=";")

    Y = np.array([2*int(y == "yes") - 1 for y in df["y"]])
    Z = np.logical_and(df["age"].values <= 60,
                       df["age"].values >= 25).astype(int)

    col_quanti = ["duration", "campaign", "pdays", "previous",
                  "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                  "euribor3m", "nr.employed"]
    #col_quali = ["job", "education", "default", "housing", "loan", "contact",
    #             "month", "day_of_week", "poutcome"]
    col_quali = ["job", "education", "housing", "loan", "contact",
                 "month", "day_of_week", "poutcome"]

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    quali_encoder = OneHotEncoder(categories="auto", drop = 'first')
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)
    X = np.delete(X, 29, 1) # unknown 겹침 loan 과 housing 에서

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=testsize, random_state=seed)
    
    # -1, 1 label to 0, 1 label
    Y_train[Y_train == -1] = 0
    Y_test[Y_test == -1] = 0
    
    scaler = StandardScaler()
    minmaxscaler = MinMaxScaler(feature_range = (0.0, 1.0)) # we do minmax scaling
    
    X_train = scaler.fit_transform(X_train)
    if minmax : 
        X_train = minmaxscaler.fit_transform(X_train) # we do minmax scaling
    
    X_test = scaler.transform(X_test)
    if minmax : 
        X_test = minmaxscaler.transform(X_test) # we do minmax scaling
        
    XZ_train = np.concatenate([X_train, Z_train.reshape(Z_train.shape[0], 1)], axis = 1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(Z_test.shape[0], 1)], axis = 1)

    return (XZ_train, X_train, Y_train, Z_train), (XZ_test, X_test, Y_test, Z_test)


""" Law school dataset """

def load_law_dataset(seed = 2021, testsize = 0.20) :

    df = pd.read_csv('datasets/law/law_data.csv', index_col=0)
    Y = np.array([int(y == "Passed") for y in df["pass_bar"]])
    Z = np.array([int(z == "White") for z in df["race"]])
    col_quanti = ['zfygpa', 'zgpa', 'DOB_yr', 'cluster_tier', 'family_income',
            'lsat', 'ugpa', 'weighted_lsat_ugpa']
    col_quali = ['isPartTime', 'sex']

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    quali_encoder = OneHotEncoder(categories="auto", drop = 'first')
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    y0_idx = np.where(Y==0)[0]
    y1_idx = np.where(Y==1)[0]

    y0_train_idx, y0_test_idx = train_test_split(y0_idx, test_size=testsize, random_state=seed)
    y1_train_idx, y1_test_idx = train_test_split(y1_idx, test_size=testsize, random_state=seed)

    train_idx = np.concatenate((y0_train_idx, y1_train_idx))                                
    test_idx = np.concatenate((y0_test_idx, y1_test_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    Z_train = Z[train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Z_test = Z[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    XZ_train = np.concatenate([X_train, Z_train.reshape(Z_train.shape[0], 1)], axis = 1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(Z_test.shape[0], 1)], axis = 1)

    return (XZ_train, X_train, Y_train, Z_train), (XZ_test, X_test, Y_test, Z_test)



def flip_sen_datasets(XS) :

    sen_idx = XS.shape[1] - 1

    XS_first = XS.clone()
    XS_first[:, sen_idx] = 1

    XS_second = XS.clone()
    XS_second[:, sen_idx] = 0

    first_set, second_set = TensorDataset(XS_first), TensorDataset(XS_second)

    return first_set, second_set, XS_first, XS_second


