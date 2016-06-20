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
import joblib
import datetime
import sys
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


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
    chunk['bookmonth'] = chunk['date_time'].apply(lambda x: x.month)
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

    
def stratshufspl(chunk, fraction, key):
    '''
    Startified shuffle split of chunks.
    '''
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
        chunk = stratshufspl(chunk, fraction, key)
        curr = formatdata(chunk)
        chunks.append(curr)
        
    # concatenate chunks
    train = pd.concat(chunks, axis=0)
    
    # split concatenated set into X and y for ml model fitting
    X = train.drop(key, axis=1, inplace=False)
    y = train[key]
    return X, y


### Make predictions
def predtest(testiter, clf, enc):
    '''
    Test file chunk size should work for 16GB memory.
    Requires formatdata from this module.
    '''
    
    # process each chunk then append at end
    first = True
    for chunk in testiter:
        X = enc.transform(formatdata(chunk))
        pred = clf.predict_proba(X)
        tmp = pd.DataFrame(pred).apply(logtorec, axis=1)
        # create series first time around, else append to existing series
        if first:
            serpred = tmp.copy()
            first = False
        else:
            serpred = serpred.append(tmp)
    # reset index
    serpred.reset_index(inplace=True, drop=True)
    return serpred


#--------------------------------------------------------------------------------------
if __name__ == '__main__':
    # set timestamp from utc to current time zone
    nowpt = datetime.datetime.now()-datetime.timedelta(hours=7)
    nowstr = nowpt.strftime('%Y%m%d_%H%M')

    # columns to read in from data that are passed to CSV reader iterators
    # final value is y-data (key)
    rawcols = ['site_name', 'user_location_city', 'user_location_region', 
                'is_package', 'srch_adults_cnt', 'srch_children_cnt', 
                'srch_destination_type_id', 'orig_destination_distance', 'hotel_country', 
                'hotel_market', 'srch_ci', 'srch_co', 'date_time', 'hotel_cluster']

    # create training data formatting iterator and warm start iterator
    trainiter = pd.read_csv('train.csv.gz', sep=',', compression='gzip', 
                            chunksize=5000000, usecols=rawcols)
    testiter = pd.read_csv('test.csv.gz', sep=',', compression='gzip', 
                           chunksize=50000, usecols=rawcols[:-1])

    # load given fraction of data and format data
    fraction = 0.04
    X, y = fractionate(trainiter, fraction, rawcols[-1])
    
    # convert categorical variables and create one hot encoder
    notcats = ['orig_destination_distance', 'stay', 'is_package']
    cats = list(X.columns)
    for cat in notcats:
        cats.remove(cat)
    # print('')
    # print('expected number of columns in sparse matrix:', 
    #       X[cats].max().sum()+len(X.columns))
    
    # create ordered dict to indicate categorical variables
    catdict = OrderedDict()
    for col in X.columns:
        if col in notcats:
            val = False
        else:
            val = True
        catdict.update({col:val})
    
    # max features comes from test data
    maxfeatures = [54, 240, 1028, 10, 10, 10, 213, 2118, 13, 367]

    # use one hot encoder to create sparse data
    mask = np.array(list(catdict.values()))
    enc = OneHotEncoder(n_values=maxfeatures, categorical_features=mask, 
                        dtype=int, sparse=True)
    Xsparse = enc.fit_transform(X.values)
    print('')
    print('Sparse matrix shape:', Xsparse.shape)
    
    # set parameters for model training
    # rf memory need ~ 2 * n_jobs * sizeof(X) 
    # memory consumption linear with number of features
    # gbcparams = {'random_state': 42 , 'n_estimators': 100, 'max_depth': 2}
    rfparams = {'random_state': 42, 'n_jobs': -1,  'class_weight': 'balanced',
    'max_features': 0.2, 'n_estimators': 125 }
    gridparams = {'max_depth': [16,20] }

    # create estimators
    # estimator = GradientBoostingClassifier(**gbcparams)
    estimator = RandomForestClassifier(**rfparams)
    grid = GridSearchCV(estimator, param_grid=gridparams, scoring='log_loss')
    # grid.set_params(n_jobs=3)
   
    # run fit on grid object
    print('')
    print('grid search for best hyperparameters on {:.2%} of data...'.format(fraction))
    grid.fit(Xsparse, y)
    
    # print scores to screen
    print('best estimator:', grid.best_estimator_)
    print('all scores:')
    for item in grid.grid_scores_:
        print(item)
    
    # pickle model in case of runtime errors during prediction
    # joblib.dump(grid, 'dump/expedia'+nowstr+'.pkl', compress=False)

    # clear memory
    X = None
    y = None
    
    # process test data by chunk and output to csv
    print('')
    print('making predictions from test data...')
    serpred = predtest(testiter, grid, enc)
    serpred.name = rawcols[-1]
    serpred.to_csv('mloutput'+nowstr+'.csv', sep=',', 
    index_label='id', header=True)
    print('model and predictions saved with suffix:',nowstr)
