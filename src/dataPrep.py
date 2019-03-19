#!/bin/env python3
"""
Load and prepare dataset for training
"""
from collections import deque
import glob
import numpy as np
import ntpath
import pandas as pd
import random
from sklearn import preprocessing

TARGET = "hs300"
RETRO_LEN = 20
CLASS_PCT = 0.01
NUM_CLASS = 3
VALIDATION_SET_PCT = 0.05
PREDIC_DAYS = 1

COL_NAMES = ('open', 
            'high',
            'low',
            'close',)
# ignore imcomplete column 'volume')

class DataSet:
    """
    load data via pandas
    preparation: normalization
    """

    def __init__(self,
            headers = None,
            names = None,
            usecols = None,
            index_col = 'time',
            skiprows = 0):
        """
        initialize with datafile parse parameters
        """
        self.headers = headers
        self.fieldnames = names
        self.index_col = index_col
        self.usecols = usecols
        self.skiprows = skiprows
        self.df = None
        self.ds = {'train': pd.DataFrame(columns=('result', 'input')),
                   'valid': pd.DataFrame(columns=('result', 'input'))}

        
    def readExcelFile(self, path, name):
        """
        read data from excel file
        """
        df = pd.read_excel(path,
                           names=[self.index_col] + [f'{name}_{fn}' for fn in self.fieldnames],
                           usecols=self.usecols,
                           skiprows=self.skiprows,
                           headers=self.headers,
                           parse_dates=True)
        df.set_index(self.index_col, inplace=True) 
        #df.fillna(method="ffill", inplace=True)
        df.dropna(inplace=True)
        if (self.df is None):
            self.df = df
        else:
            self.df = self.df.join(df)

    def preprocess(self):
        # normalization with "from sklearn import preprocessing"
        #scaler = preprocessing.StandardScaler().fit(self.df.values)
        #scaler = preprocessing.MinMaxScaler().fit(self.df.values)
        #self.df = pd.DataFrame(scaler.transform(self.df.values), columns=self.df.columns, index=self.df.index)
        for col in self.df.columns:
            # print inif values for debugging purpose
            #print(col, self.df[self.df[col] == np.inf])
            if (col != f'{TARGET}_close'):
                self.df[col] = self.df[col].pct_change()
                self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
                self.df.dropna(inplace=True)
                self.df[col] = preprocessing.scale(self.df[col].values) # scale between 0 and 1
        # calculate diff between close price of today and tomorrow
        self.df['target'] = self.df[f'{TARGET}_close'].shift(-PREDIC_DAYS)
        self.df['target'] = list(map(classifyDiff, self.df[f'{TARGET}_close'], self.df['target']))
        self.df = self.df.drop(f'{TARGET}_close', 1)  # already classified to 'target'
        self.df.dropna(inplace=True)
        # split validation dataset
        vdSize = int(len(self.df)*VALIDATION_SET_PCT)
        # generate hist retro sequences
        self.retroHist(vdSize)

    def retroHist(self, vdSize):
        # sort data according to classification
        # and split in training and validation data set
        leftlen = len(self.df)
        retro = deque(maxlen=RETRO_LEN)
        for i in self.df.values:
            retro.append(i[:-1])
            leftlen -= 1
            if (len(retro) == retro.maxlen):
                if (leftlen > vdSize):
                    self.ds['train'] = self.ds['train'].append({'result' : i[-1], 'input' : np.array(retro)}, ignore_index=True)
                else:
                    self.ds['valid'] = self.ds['valid'].append({'result' : i[-1], 'input' : np.array(retro)}, ignore_index=True)
        print(self.ds['train'].groupby('result').size().reset_index(name='counts'))
        print(self.ds['valid'].groupby('result').size().reset_index(name='counts'))

    def getDataSets(self, tn, vn):
        dt = self.ds['train'].groupby('result',group_keys=False).apply(lambda x: x.sample(tn, random_state=1)).sample(frac=1)
        dv = self.ds['valid'].groupby('result',group_keys=False).apply(lambda x: x.sample(vn, random_state=1)).sample(frac=1)
        return np.stack(dt['input'].values), dt['result'].values, np.stack(dv['input'].values), dv['result'].values

def classify(diff):
    if (diff > CLASS_PCT):
        return int(2)
    elif (diff < -CLASS_PCT):
        return int(0)
    else:
        return int(1)

def classifyDiff(current, future):
    if current == 0:
        return classify(0)
    diff = ((float(future) - float(current)))/float(current)
    return classify(diff)

def readDataFromFile(files):
    ds = DataSet(index_col = 'time',
            names = COL_NAMES,
            usecols = 'c:g',
            # ignore imcomplete column 'volume' usecols = 'c:g,i',
            skiprows = 1)
    for f in files:
        # take file name as dataset label
        name = ntpath.basename(f)
        ds.readExcelFile(f, name[:name.index('.')])
    ds.preprocess()
    return ds

if __name__ == '__main__':
    data_files = 'data/*.xlsx'
    ds = readDataFromFile(glob.glob(data_files))
    train_data, train_label, valid_data, valid_label = ds.getDataSets(10, 2)
    print(train_data[0])
    print(train_data[-1])
    print(len(train_data))
    print(len(valid_data))

