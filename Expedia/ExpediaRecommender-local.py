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
import sys
import math
import multiprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV


### Utility functions
def logtorec(row):
    '''convert probabilities for 100 classes to top 5 predictions'''
    ypred = list(zip(row, list(range(100))))
    ypred5 = ' '.join([str(t[1]) for t in sorted(ypred, reverse=True)][:5])
    return ypred5


def formatdata(chunk):
    '''Format data of each startified chunk, drop columns that are not needed.'''
    
    # fill NAs for origin-destination distance
    chunk['orig_destination_distance'].fillna(0.0, inplace=True)
  
    # parse datetime objects
    for col in ['date_time', 'srch_ci', 'srch_co']:
        chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
    # create booking dayofyear
    chunk['bookdoy'] = chunk['date_time'].apply(lambda x: x.dayofyear)
    # fill NA values with booking datetime and stay length of zero
    chunk['srch_co'].fillna(chunk['date_time'], inplace=True)
    chunk['srch_ci'].fillna(chunk['date_time'], inplace=True)
    # create stay length
    chunk['stay'] = chunk['srch_co']-chunk['srch_ci']
    chunk['stay'] = chunk['stay'].apply(lambda x: x.days)

    # create stay dayofyear
    chunk['staydoy'] = chunk['srch_ci'].apply(lambda x: x.dayofyear)
     
    # drop processed columns
    chunk.drop(['date_time','srch_ci','srch_co'], axis=1, inplace=True)
    
    return chunk

    
def sparsify(dfdum, dumcols=[], trainsparse=False):
    '''Needs to have two branches, one for train and one for test, 
       both cannot be created at same time due to memory issues.
       Sparsify dataset of all categorical variables if desired.'''

    # create dummies
    # dfdum = pd.get_dummies(dfdum).astype(int)

    # output training column names and calculate sparsity if necessary
    if len(dumcols) == 0:   # meaning this is training data
        print('')
        print('sparsifying data if necessary...')
        outcols = dfdum.columns
        # calculate sparsity ratio if training data
        sparsity_ratio = lambda X: 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])
        spratio = sparsity_ratio(dfdum.values)
        print('')
        print("sparsity ratio:", spratio)
        # sparsify if threshold is met
        if spratio >= 0.9 and trainsparse:
            trainsparse = True
            dfdum = sparse.csr_matrix(dfdum)
        print('data format:',type(dfdum))
        print('')
        print('number of dummy features:', len(outcols))
        # return data, list of dummy column names and sparsity flag
        return dfdum, outcols, trainsparse

    # process test data, check for missing columns and add them as all zeros
    else:
        if len(dfdum.columns) == len(dumcols) and all(dfdum.columns == dumcols):
            pass
        else:
            missing = set(dumcols).difference(set(dfdum.columns))
            for col in missing:
                dfdum[col] = np.zeros((len(dfdum),1),dtype=int)
            dfdum = dfdum[list(dumcols)]
        if trainsparse:
            dfdum = sparse.csr_matrix(dfdum)
        # return the test data with dummy columns if necessary
        return dfdum


def stshsp(chunk, fraction, key):
    '''Startified shuffle split of chunks.'''
    sss = StratifiedShuffleSplit(chunk[key], test_size=fraction, 
                                 random_state=42, n_iter=1)
    for _, idx in sss:
        train = chunk.iloc[idx].copy()
    return train


def fractionate(trainiter, fraction, key):
    '''Utilizes only one core and <10% of 16GB.'''
    print('')
    print('loading data...')
    
    # create empty list and add formatted data chunks
    chunks = list()
    for chunk in trainiter:
        chunk = stshsp(chunk, fraction, key)
        curr = formatdata(chunk)
        chunks.append(curr)
        
    # concatenate chunks
    train = pd.concat(chunks, axis=0)
    
    # split concatenated set into X and y for ml model fitting
    X = train.drop(key, axis=1, inplace=False)
    y = train[key]
    return X, y


### Training and testing functions
def warmstart(trainiter, estimator, fraction, dumcols=[], trainsparse=False):
    '''train model in chunks using warmstart'''

    # set warm start to True
    estimator.set_params(warm_start=True)

    # get initial number of estimators to serve as increment
    inc = estimator.get_params()['n_estimators']
    modfit = False
    # tmpcols = []
    # first = True
    counter = 1
    for chunk in trainiter:
        chunkfrac = stshsp(chunk, fraction, rawcols[-1])
        train = formatdata(chunkfrac)
#         sss = StratifiedShuffleSplit(chunk[key], test_size=fraction, 
#                                      random_state=42, n_iter=1)
#         for _, idx in sss:
#             train = chunk.iloc[idx] # view
#             yc = chunk[key].iloc[idx].copy() # copy
        # make sure number of labels is same for each chunk
        # if current chunk is different, disregard it
        if counter > 1:
            if len(yc.unique()) < uniques:
                continue
        Xtrain = train.drop(rawcols[-1], axis=1, inplace=False) # copy
        yc = train[rawcols[-1]]
        X = sparsify(Xtrain, [1], trainsparse)
        Xc = X.values.copy(order='C')
        # print(str(len(yc.unique())))
        # sys.stdout.write(' ')
        estimator.fit(Xc,yc)
        nprev = estimator.get_params()['n_estimators']
        print('total number of estimators:',nprev)
        estimator.set_params(n_estimators = nprev+inc)
        modfit = True
        counter += 1
        uniques = len(yc.unique())
    print('')
    print('model fitted:', modfit)
    return estimator


def gridtrainfraction(trainiters, estimator, grid, gridfrac, trainfrac):
    ''' 
    Runs grid search cv with given estimator on portion of training data.
    Read in data once for grid search, clear, then fit model with best params.
    '''

    # gbc does not work with sparse data so check for random forest estimator
    # if random forest is used here, then check if sparsify is useful 
    # if 'class_weight' in estimator.get_params():
    #     ts = True
    # else:
    #     ts = False
    
    # retrieve fraction of data
    Xtrain, y = fractionate(trainiters[0], gridfrac, rawcols[-1])
    
    # create dummy variables and determine if data should be sparsified
    # dc = []
    # X, dumcols, trainsparse = sparsify(Xtrain, dc, ts)
    print('')
    print('grid search for best hyperparameters on {:.2%} of data...'.format(gridfrac))
    
    # run fit on passed grid object
    grid.fit(Xtrain, y)
    
    # print scores to screen
    print('best parameters:', grid.best_params_)
    print('all scores:')
    for item in grid.grid_scores_:
        print(item)
    
    # clear memory
    Xtrain = None
    X = None
    y = None
    
    # train on larger dataset using grid params and warm start if necessary
    estimator.set_params(**grid.best_params_)
    
    # if training with warmstart
    # clfws = warmstart(trainiters[1], estimator, trainfrac, dumcols, trainsparse)
    # return clfws, dumcols, trainsparse
    
    # if no warmstart
    # retrieve new fraction of data
    Xtrain, y = fractionate(trainiters[1], trainfrac, rawcols[-1])
    
    # create dummy variables and determine if data should be sparsified
    # dc = []
    # X, dumcols, trainsparse = sparsify(Xtrain, dc, ts)

    # fit estimator and return
    print('')
    print('training model on {:.2%} of data...'.format(trainfrac))
    estimator.fit(Xtrain,y)
    dumcols = [1]
    trainsparse = False
    return estimator, dumcols, trainsparse


def trainfraction(trainiters, estimator, trainfrac):
    '''fraction of whole dataset, use fast computational method and dummy features'''

    # retrieve data, determine sparcity and dummy column names
    # also check if estimator is gbc or rf
    Xtrain, y = fractionate(trainiters[0], trainfrac, rawcols[-1])
    # dc = []
    # if 'class_weight' in estimator.get_params():
    #     ts = True
    # else:
    #     ts = False
    # _, dumcols, trainsparse = sparsify(Xtrain, dc, ts)
    # train rf model
    print('')
    print('training model on {:.2%} of data...'.format(trainfrac))
    clf = estimator
    clf.fit(Xtrain, y)
    # clf = warmstart(trainiters[1], estimator, trainfrac, [], False)
    dumcols = [1]
    trainsparse = False
    return clf, dumcols, trainsparse


### Make predictions
def predtest(testiter, clf, rawcols, dumcols, trainsparse):
    '''Test file chunk size optimized for 16GB memory'''

    # class is not present in test data so remove from column list
    rawcols.remove('hotel_cluster')
    
    # process each chunk then append at end
    first = True
    for chunk in testiter:
        X = formatdata(chunk)
        # X = sparsify(chunk, dumcols, trainsparse)
        pred = clf.predict_proba(X)
        tmp = pd.DataFrame(pred).apply(logtorec, axis=1)
        if first:
            serpred = tmp.copy()
            first = False
        else:
            serpred = serpred.append(tmp)
    serpred.reset_index(inplace=True, drop=True)
    return serpred


#--------------------------------------------------------------------------------------
if __name__ == '__main__':
    # set timestamp from utc to current time zone
    nowpt = datetime.datetime.now()#-datetime.timedelta(hours=7)
    nowstr = nowpt.strftime('%Y%m%d_%H%M')

    # columns to read in from data that are passed to CSV reader iterators
    # final value is y-data (key)
    rawcols = ['site_name', 'user_location_country', 'user_location_region', 
                'is_package', 'srch_adults_cnt', 'srch_children_cnt', 
                'srch_destination_type_id', 'orig_destination_distance', 'hotel_country', 
                'hotel_market', 'srch_ci', 'srch_co', 'date_time', 'hotel_cluster']
    dtypes = dict(zip(rawcols,['int']*len(rawcols)))

    # create training data formatting iterator and warm start iterator
    dataurl = '/Users/dbricare/Documents/Python/datasets/expedia/'
    trainiter1 = pd.read_csv(dataurl+'train.csv.gz', sep=',', compression='gzip', 
                            chunksize=5000000, usecols=rawcols)
    trainiter2 = pd.read_csv(dataurl+'train.csv.gz', sep=',', compression='gzip', 
                            chunksize=5000000, usecols=rawcols)
    # create CSV reader iterator for test data
    testiter = pd.read_csv(dataurl+'test.csv.gz', sep=',', compression='gzip', 
                           chunksize=20000, usecols=rawcols)
    
    # set parameters for model training
    # read in classweights and convert to dictionary
    # dfweights = pd.read_csv('classweights.csv', sep=',', header=None)
    # dfweights.set_index(0, inplace=True)
    # classweights = dfweights[1].to_dict()
    # rfparams = {'random_state': 42, 'n_jobs': -1,  'class_weight': classweights,
    #             'n_estimators': 300, 'max_depth': 14 }
    gbcparams = {'random_state': 42 }
    gridparams = {'n_estimators': [50,100], 'max_depth': [2,4,6]}

    # create estimators
    estimator = GradientBoostingClassifier(**gbcparams)
#     estimator = RandomForestClassifier(**rfparams)
    grid = GridSearchCV(estimator, param_grid=gridparams, scoring='log_loss')
    grid.set_params(n_jobs=3)

    # train fraction of data
    # clf, dumcols, trainsparse = trainfraction([trainiter1,trainiter2], 
    # estimator, trainfrac=0.02)
    clf, dumcols, trainsparse = gridtrainfraction([trainiter1, trainiter2], 
    estimator, grid, gridfrac=0.002, trainfrac=0.02)

    # pickle model in case of runtime errors during prediction
    # joblib.dump(clf, 'expedia'+nowstr+'.pkl', compress=True)

    # pickle scores as well (just for curiosity)
    # joblib.dump(grid.grid_scores_, 'scores'+nowstr+'.pkl', compress=True)

    # process test data and output to csv
    print('')
    print('making predictions from test data...')
    serpred = predtest(testiter, clf, rawcols, dumcols, trainsparse)
    serpred.name = 'hotel_cluster'
    serpred.to_csv('mloutput'+nowstr+'.csv', sep=',', 
    index_label='id', header=True)
    print('model and predictions saved with suffix:',nowstr)
