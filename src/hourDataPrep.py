#!/bin/env python3

from collections import deque
import datetime
import numpy as np
import ntpath
import pandas as pd
import random
import sys

class HourDataPrep:
    def __init__(self, retroLen=32, classPcnt=0.008, ahead=4):
        self.retroLen = retroLen
        self.classPcnt = classPcnt
        self.ahead = ahead

    def readData(self, fpath, cols):
        df = pd.read_excel(fpath,
                        header=0,
                        index_col=0,
                        usecols=cols,
                        parse_dates=True)
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        # separate open transaction data
        dfo = df.at_time('09:30')
        df = df.between_time('10:00','15:00')
        # put open price as pre_close if not available
        dfo['pre_close'].fillna(dfo['close'])
        # set close price of previous date as pre_close
        for i in dfo.index:
            try:
                df.loc[i.isoformat(), 'pre_close'] = dfo.loc[i, 'pre_close']
            except KeyError:
                print("entry not found ",i)
                continue
        # calculate percent change of close prices between this and next entry
        df['nxt_close'] = df['close'].shift(-self.ahead)
        df.dropna(inplace=True)
        df['close_chg'] = df['nxt_close']/df['close'] - 1
        # classify close price percent change
        df.loc[df['close_chg'] <= -self.classPcnt, 'class'] = 0
        df.loc[df['close_chg'].between(-self.classPcnt, self.classPcnt, inclusive=False), 'class'] = 1
        df.loc[df['close_chg'] >= self.classPcnt, 'class'] = 2
        # normalize price to percent change to pre_close
        for col in df.columns[:4]:
            df[col] = df[col]/df['pre_close'] - 1
        # normalize vol to percent change to previouse entry
        df['vol'] = df['vol'].pct_change()
        df['vol'].fillna(1,inplace=True)
        # drop rows that don't have volume
        #df = df[df.vol != 0]
        df.dropna(inplace=True)

        # create hist data frame
        histDf = pd.DataFrame(columns=('time','data'))
        histDf.set_index('time', inplace=True)
        # put last 64 entries to hist array
        retro = deque(maxlen=self.retroLen)
        for r in df.filter(items=df.columns[:5]).iterrows():
            retro.append(list(r[1]))
            if (len(retro) == retro.maxlen):
                histDf.loc[r[0]] = {'data' : np.array(retro)}
        # remove data without hist
        df.drop(df.index.difference(histDf.index), inplace=True)
        print(df.info(), histDf.info())
        print(histDf.tail(1).values[0])
        print(df.head(), df.tail())
        self.df = df
        self.histDf = histDf

    def getDataSets(self, n, date=None, dclass=None):
        df = self.df
        if n > 0:
            df = df.loc[pd.to_datetime(date):]
        elif n < 0:
            df = df.loc[:pd.to_datetime(date)]
        if (dclass is not None):
            df = df.loc[df['class'] == dclass]
        df = self.histDf.join(df['class'])
        df.dropna(inplace=True)
        print(df.groupby('class').count())
        df = df.sample(abs(n), random_state=1)
        print(f'time range: {df.index.min()} ~ {df.index.max()} -- {len(df.index)}')
        print(df.head(), df.tail())
        return np.stack(df['data'].values), df['class'].values

if __name__ == '__main__':
    hdf = HourDataPrep()
    hdf.readData('data/hs300_hours.xlsx', 'b:g,j')
    hdf.getDataSets(128, '2019-03-08')
    hdf.getDataSets(-9856,'2019-03-08')
    hdf.getDataSets(32,'2019-03-08', dclass=2)
