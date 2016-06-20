'''
Train and test machine learning model for 2016 Kaggle Expedia competition.

Be sure train and test data is copied over from s3.

Possible strategies:
1. Grid search and create model from random subset of data
2. Load data iteratively, train each chunk with warmstart, no parameter tuning
3. Grid search random subset of model then fit whole dataset with warmset
4. Create model from random subsample of data
'''

import pandas as pd
import numpy as np
from scipy import sparse
import joblib
import datetime
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV


### Utility functions
def logtorec(row):
    '''convert probabilities to top 5 predictions'''
    ypred = list(zip(row, list(range(100))))
    ypred5 = ' '.join([str(t[1]) for t in sorted(ypred, reverse=True)][:5])
    return ypred5

    
def sparsify(dfdum, traincols=[], trainsparse=False):
    '''Needs to have two branches, one for train and one for test, 
       both cannot be created at same time due to memory issues.
       
       Sparsify dataset of all categorical variables.'''
    # create dummies
    dfdum = pd.get_dummies(dfdum.astype(str))
    # output training column names and calculate sparsity if necessary
    if len(traincols) == 0:
        print('')
        print('sparsifying data if necessary...')
        outcols = dfdum.columns
        trainsparse = False
        sparsity_ratio = lambda X: 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])
        spratio = sparsity_ratio(dfdum.values)
        print('')
        print("sparsity ratio:", spratio)
        if spratio >= 0.9:
            trainsparse = True
            dfdum = sparse.csr_matrix(dfdum)
        print('data format:',type(dfdum))
        return dfdum, outcols, trainsparse
    else:
        # check if test data is missing columns and add them as all zeros
        if len(dfdum.columns) == len(traincols) and all(dfdum.columns == traincols):
            pass
        else:
            missing = set(traincols).difference(set(dfdum.columns))
            for col in missing:
                dfdum[col] = np.zeros((len(dfdum),1),dtype=int)
            dfdum = dfdum[list(traincols)]
        if trainsparse:
            dfdum = sparse.csr_matrix(dfdum)
        return dfdum


def fractionate(trainiter, fraction):
    print('')
    print('loading data...')
    first = True
    key = 'hotel_cluster'
    for chunk in trainiter:
        sss = StratifiedShuffleSplit(chunk[key], test_size=fraction, 
                                     random_state=42, n_iter=1)
        for _, idx in sss:
            if first:
                train = chunk.iloc[idx]
                first = False
            else:
                train = train.append(chunk.iloc[idx], ignore_index=True)
    return train


### Training functions
def warmstart(trainiter, rfparams):
    '''train model in chunks using warmstart'''
    key = 'hotel_cluster'
    clf = RandomForestClassifier(n_estimators=100, warm_start=True, **rfparams)
    for chunk in trainiter:
        X = chunk.drop(key, axis=1)
        y = chunk[key]
        clf.fit(X,y)
    return clf


def trainfraction(trainiter, rfparams, fraction=0.01):
    '''fraction of whole dataset, use fast computational method and binarize features'''
    train = fractionate(trainiter, fraction)
    X_train = train.drop('hotel_cluster', axis=1)
    X, traincols, trainsparse = sparsify(X_train)
    y = train['hotel_cluster']
    # train rf model
    print('')
    print('training model...')
    clf = RandomForestClassifier(n_estimators=100, **rfparams)
    clf.fit(X, y)
    return clf, traincols, trainsparse

def testfraction(clf, datacols):
    datacols.remove('hotel_cluster')
    datacols.append('id')
    testiter = pd.read_csv('test.csv.gz', sep=',', compression='gzip', 
                           chunksize=200000, usecols=datacols)
    # process each chunk then append at end
    first = True
    for chunk in testiter:
        X_test = chunk.drop('id', axis=1)
        X = sparsify(X_test, traincols, trainsparse)
        pred = clf.predict_proba(X)
        dfpred = pd.DataFrame(pred)
        tmp = dfpred.apply(logtorec, axis=1)
        if first:
            serpred = tmp.copy()
            first = False
        else:
            serpred = serpred.append(tmp)
    serpred.reset_index(inplace=True, drop=True)
    return serpred

    
def gridtrainfraction(trainiter, rfparams):
    ''' read in data once for grid search, clear, then again for model fit'''
    train = fractionate(trainiter, fraction=0.002)
    clf = RandomForestClassifier(**rfparams)
    grid = GridSearchCV(clf, param_grid=gridparams, scoring='log_loss', n_jobs=1)
    X_train = train.drop('hotel_cluster', axis=1)
    X = sparsify(pd.get_dummies(X_train.astype(str)))
    y = train['hotel_cluster']
    grid.fit(X,y)
    
    print(grid.best_params_)
    print(grid.grid_scores_)
    
    train = None
    X_train = None
    X = None
    y = None
    clf = None
    
    train = fractionate(trainiter, fraction=0.01)
    X_train = train.drop('hotel_cluster', axis=1)
    X = sparsify(pd.get_dummies(X_train.astype(str)))
    y = train['hotel_cluster']
    bestparams = grid.best_params_
    clf = RandomForestClassifier(**rfparams)
    clf.set_params(**bestparams)
    clf.fit(X,y)
    return clf


if __name__ == '__main__':
    # set timestamp
    now = datetime.datetime.now()
    nowstr = now.strftime('%Y%m%d_%H%M')

    # columns to read in for data
    datacols = ['site_name', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 
                'srch_children_cnt', 'srch_destination_type_id', 'hotel_continent', 
                'hotel_cluster']

    # create training data iterator
    # print('loading training data...')
    trainiter = pd.read_csv('train.csv.gz', sep=',', compression='gzip', 
                            chunksize=1000000, usecols=datacols)
        
    # train model
    rfparams = {'random_state' : 42, 'n_jobs' : 4,  'class_weight' : 'balanced'}
    gbcparams = {'random_state' : 42, 'subsample' : 0.5}
    gridparams = {'max_depth' : [2,4], 'n_estimators' : [75,100]}

    # train fraction of data and pickle model
    clf, traincols, trainsparse = trainfraction(trainiter, rfparams, fraction=0.001)
    joblib.dump(clf, 'gbc'+nowstr+'.pkl', compress=True)

    # free up memory
#     trainiter = None

    # pickle scores as well (just for curiosity)
    # joblib.dump(grid.grid_scores_, 'scores'+nowstr+'.pkl', compress=True)

    # load test data and format output for csv
    print('')
    print('making predictions from test data...')
    serpred = testfraction(clf, datacols)
    serpred.name = 'hotel_cluster'
    serpred.to_csv('mloutput'+nowstr+'.csv', sep=',', 
    index_label='id', header=True)
