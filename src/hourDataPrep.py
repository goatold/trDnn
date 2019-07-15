#!/bin/env python3

from collections import deque
import datetime
import numpy as np
import ntpath
import pandas as pd
import random
import sys
from sklearn import preprocessing

precBasedCols = [
    'open', 'high', 'low',
    'open_d', 'close_d', 'high_d', 'low_d',
    'open_w', 'close_w', 'high_w', 'low_w'
]
volCols = ['vol', 'volume_d', 'volume_w']
genCols = [
    'MA10', 'j', 'DEA', 'd', 'MA20', 'DIF', 'MACD', 'k', 'MA60', 'MA5',
    'DIF_d', 'k_d', 'MA20_d', 'MA60_d', 'd_d', 'DEA_d', 'j_d', 'MA10_d', 'MACD_d', 'MA5_d',
    'MA60_w', 'DEA_w', 'j_w', 'MA10_w', 'MA20_w', 'k_w', 'd_w', 'MA5_w', 'MACD_w', 'DIF_w'
]


class HourDataPrep:
    def __init__(self, retroLen=64, classPcnt=0.01, ahead=4):
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
        # scale suplemental data
        ################
        # for col in df.columns[5:]:
        #     df[col] = preprocessing.scale(df[col].values)
        # # separate open transaction data
        # dfo = df.at_time('09:30')
        # df = df.between_time('10:00', '15:00')
        # # put open price as pre_close if not available
        # dfo['pre_close'].fillna(dfo['close'])
        # # set close price of previous date as pre_close
        # for i in dfo.index:
        #     try:
        #         df.loc[i.isoformat(), 'pre_close'] = dfo.loc[i, 'pre_close']
        #     except KeyError:
        #         print("entry not found ", i)
        #         continue
        ################
        # calculate percent change of close prices between this and next entry
        # column to compare with pre-close c,d,e,f, v,w,x,y, al,am,an,ao
        df['nxt_close'] = df['close'].shift(-self.ahead)
        df.dropna(inplace=True)
        df['close_chg'] = df['nxt_close']/df['close'] - 1
        # classify close price percent change
        df.loc[df['close_chg'] <= -self.classPcnt, 'class'] = 0
        df.loc[df['close_chg'].between(-self.classPcnt,
                                       self.classPcnt, inclusive=False), 'class'] = 1
        df.loc[df['close_chg'] >= self.classPcnt, 'class'] = 2
        # classify close price up/down
        df.loc[df['close_chg'] <= 0, 'ud'] = 0
        df.loc[df['close_chg'] > 0, 'ud'] = 1
        # normalize price to percent change to pre_close
        for col in precBasedCols:
            df[col] = df[col]/df['pre_close'] - 1
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
        # normalize vol cols to percent change to previouse entry
        for col in volCols:
            df[col] = df[col].pct_change()
            df[col].fillna(1, inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
        for col in genCols:
            df[col] = preprocessing.scale(df[col].values)

        # create hist data frame
        histDf = pd.DataFrame(columns=('time', 'data'))
        histDf.set_index('time', inplace=True)
        # put last 64 entries to hist array
        retro = deque(maxlen=self.retroLen)
        #dataCols = df.columns.values.tolist()
        #dataCols = dataCols[:5] + dataCols[6:16]
        dataCols = precBasedCols + volCols + genCols
        print(dataCols)
        for r in df.filter(items=dataCols).iterrows():
            retro.append(list(r[1]))
            if (len(retro) == retro.maxlen):
                histDf.loc[r[0]] = {'data': np.array(retro)}
        # remove data without hist
        df.drop(df.index.difference(histDf.index), inplace=True)
        print(df.info(), histDf.info())
        print(df.head(), df.tail())
        print(
            f'time range: {df.index.min()} ~ {df.index.max()} -- {len(df.index)}')
        self.df = df
        self.histDf = histDf

    def getDataSets(self, beginDate=None, endDate=None, target=None, getUd=False):
        df = self.df
        if beginDate is not None:
            df = df.loc[pd.to_datetime(beginDate):]
        if endDate is not None:
            df = df.loc[:pd.to_datetime(endDate)]
        if getUd:
            df = self.histDf.join(df['ud'])
        else:
            df = self.histDf.join(df['class'])
        labelName = df.columns[1]
        if (target is not None):
            df = df.loc[df[labelName] == target]
        df.dropna(inplace=True)
        print(df.groupby(labelName).count())
        df = df.sample(frac=1.0, random_state=1)
        print(
            f'time range: {df.index.min()} ~ {df.index.max()} -- {len(df.index)}')
        return np.stack(df['data'].values), df[labelName].values

if __name__ == '__main__':
    hdf = HourDataPrep()
    cutDate = '2019-01-01'
    hdf.readData('data/000300hdw.xlsx', 'a,c:g,j:t,v:aj,al:az')
    hdf.getDataSets(beginDate='2015-01-01', endDate=cutDate)
    hdf.getDataSets(beginDate=cutDate)
    hdf.getDataSets(beginDate=cutDate, target=2)
    d, l = hdf.getDataSets(beginDate=cutDate, getUd=True)
    print(d[0], l[0])
