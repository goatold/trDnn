#!/bin/env python3
"""
Load and prepare dataset for training
"""
from collections import deque
import datetime
import glob
import numpy as np
import ntpath
import pandas as pd
import random
#from sklearn import preprocessing

import conf

class DataSet:
    """
    load data via pandas
    preparation: normalization
    """

    def __init__(self,
            colNames,
            index_col = 0,
            base_col = -2,
            target_col = -2,
            skiprows = 1):
        """
        initialize with datafile parse parameters
        """
        self.colNames = colNames
        self.index_col = colNames[index_col]
        self.base_col = colNames[base_col]
        self.target_col = colNames[target_col]
        self.skiprows = skiprows
        self.rDfs = {}

        
    def readExcelFile(self, path, table, cols):
        """
        read data from excel file
        """
        df = pd.read_excel(path,
                           names=self.colNames,
                           index_col=self.index_col,
                           usecols=cols,
                           skiprows=self.skiprows,
                           parse_dates=True)
        df.dropna(inplace=True)
        # drop rows that don't have volume
        df = df[df.volume != 0]
        self.rDfs[table] = df

    def getTargetData(self, table, pred):
        df = self.rDfs[table]
        self.targetDf = pd.DataFrame(df[self.target_col].shift(-pred), index=df.index)
        # calculate target column
        self.targetDf['target'] = list(map(classifyDiff, df[self.target_col], self.targetDf[self.target_col]))
        self.targetDf.dropna(inplace=True)
        print(self.targetDf.head(3))
        print(self.targetDf.tail(3))
        print(f'target: {self.targetDf.index.min()} ~ {self.targetDf.index.max()} -- {len(self.targetDf.index)}')

    def preprocess(self):
        for df in self.rDfs.values():
            df['volume'] = df['volume'].pct_change()
            # take value of next row as ratio base
            df['base'] = df[self.base_col].shift(1)
            df.dropna(inplace=True)
            # assume column 'volumn' is the second last
            for col in df.columns[:-2]:
                # in/decrease ratio
                df[col] = df[col]/df['base'] - 1
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)
            df.drop(columns=['base'], inplace=True)

    def retroHist(self, retroLen, kTable):
        # join all tables
        df = self.rDfs[kTable]
        for t in self.rDfs.keys():
            if t == kTable:
                continue
            df = df.join(self.rDfs[t], sort=True, rsuffix=f'_{t}')
            df.dropna(inplace=True)
        retro = deque(maxlen=retroLen)
        histDf = pd.DataFrame(columns=('time','data'))
        histDf.set_index('time', inplace=True)
        for r in df.iterrows():
            retro.append(list(r[1]))
            if (len(retro) == retro.maxlen):
                histDf.loc[r[0]] = {'data' : np.array(retro)}
        self.histDf = histDf.sort_index()
        print(f'hist: {histDf.index.min()} ~ {histDf.index.max()} -- {len(histDf.index)}')

    def getHist(self, n=1, date=None):
        if n > 0:
            # first n records after date inclusive
            return self.histDf.loc[pd.to_datetime(date):].first(f'{n}d')
        elif n < 0:
            # last n records before date inclusive
            return self.histDf.loc[:pd.to_datetime(date)].last(f'{-n}d')
        else:
            return self.histDf

    def getDataSets(self, n, date=None):
        df = self.histDf
        if n > 0:
            df = self.histDf.loc[pd.to_datetime(date):].sample(n, random_state=1)
        elif n < 0:
            df = self.histDf.loc[:pd.to_datetime(date)].sample(-n, random_state=1)
        df = df.join(self.targetDf)
        df.dropna(inplace=True)
        print(f'time range: {df.index.min()} ~ {df.index.max()} -- {len(df.index)}')
        print(df.head(3), df.tail(3))
        return np.stack(df['data'].values), df['target'].values

def classify(diff):
    if (diff > conf.CLASS_PCT):
        return int(2)
    elif (diff < -conf.CLASS_PCT):
        return int(0)
    else:
        return int(1)

def classifyDiff(current, future):
    if current == 0:
        return classify(0)
    diff = ((float(future) - float(current)))/float(current)
    return classify(diff)

def readDataFromFile(files, colNames, cols):
    ds = DataSet(colNames)
    for f in files:
        # take file name as dataset label
        name = ntpath.basename(f)
        ds.readExcelFile(f, name[:name.index('.')], cols)
    ds.getTargetData(conf.TARGET_TABLE, 1)
    ds.preprocess()
    ds.retroHist(conf.RETRO_LEN, conf.TARGET_TABLE)
    # remove defact data
    ds.targetDf.drop(ds.targetDf.index.difference(ds.histDf.index), inplace=True)
    ds.histDf.drop(ds.histDf.index.difference(ds.targetDf.index), inplace=True)
    return ds

if __name__ == '__main__':
    data_files = 'data/*.xlsx'
    ds = readDataFromFile(glob.glob(data_files), conf.COL_NAMES, conf.EXCEL_COL_TO_READ)
    sdate = ds.histDf.index[-128]
    train_data, train_label = ds.getDataSets(-3, sdate - datetime.timedelta(days=1))
    valid_data, valid_label = ds.getDataSets(3, sdate)
    print(sdate)
    print(train_label)
    #print(valid_data)
    print(valid_label)

