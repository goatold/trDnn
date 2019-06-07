#!/bin/env python3
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint

import dataPrep
import hourDataPrep
import conf

def createSeqMod(inputShape, outputSize=2):
    # create sequential model with m LSTM layers and n Dense layers
    # we may replace LSTM with CuDNNLSTM for GPU
    model = Sequential()
    for i in range(conf.NUM_OF_LSTM):
        if i == 0:
            model.add(LSTM(conf.lstmOutSize, input_shape=(inputShape), return_sequences=True))
        if i == conf.NUM_OF_LSTM-1:
            model.add(LSTM(conf.lstmOutSize))
        else:
            model.add(LSTM(conf.lstmOutSize, return_sequences=True))
        # helps prevent overfitting
        model.add(Dropout(rate=conf.dropRatio))
        model.add(BatchNormalization())

    for i in range(conf.NUM_OF_DENSE-1):
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(rate=conf.dropRatio))

    model.add(Dense(outputSize, activation='softmax'))


    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return model

def trainModel(df, cutDate, binResult=False):
    tensorboard = TensorBoard(log_dir="logs/{}".format(conf.LOG_NAME))

    # unique file name that will include the epoch and the validation acc for that epoch
    modelpath = f"models/retro{df.retroLen}-clsPcnt{df.classPcnt}-nxt{df.ahead}.model"
    # saves only the best ones
    checkpoint = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    valid_data, valid_label = df.getDataSets(beginDate = cutDate, getUd = binResult)
    train_data, train_label = df.getDataSets(endDate = cutDate, getUd = binResult)
    labelSize = valid_label.max()+1

    model = createSeqMod(train_data.shape[1:], labelSize)

    # Train model
    model.fit(
        train_data, train_label,
        batch_size=conf.BATCH_SIZE,
        epochs=conf.EPOCHS,
        validation_data=(valid_data, valid_label),
        callbacks=[tensorboard, checkpoint],
    )
    # Save model
    #model.save("models/{}".format(conf.LOG_NAME))
    # load model with best val_acc
    model = load_model(modelpath)
    # model.summary()
    # print(model.input_shape)
    # score model with specific result class
    for i in range(int(labelSize)):
        valid_data, valid_label = hdf.getDataSets(beginDate = cutDate, target=i)
        score = model.evaluate(valid_data, valid_label, verbose=0)
        print(f'Test{i} loss: {score[0]} accuracy: {score[1]}')
    return model

if __name__ == '__main__':
    #ds = dataPrep.readDataFromFile(glob.glob(conf.data_files), conf.COL_NAMES, conf.EXCEL_COL_TO_READ)
    #train_data, train_label = ds.getDataSets(-conf.sampleSizeT, conf.trainDataBefore)
    #valid_data, valid_label = ds.getDataSets(conf.sampleSizeV, conf.validDataAfter)
    # consider keras.utils.to_categorical(label, num_classes=NUM_CLASS)

    hdf = hourDataPrep.HourDataPrep(retroLen=128, classPcnt=0.008, ahead=8)
    hdf.readData('data/hs300_hours_tech.xlsx', 'c:h,k:u')
    cutDate = '2019-01-01'

    model = trainModel(hdf, cutDate)
    # try predict with last data entry
    print(model.predict(np.stack(hdf.histDf.tail(1).values[0])))
    model = trainModel(hdf, cutDate, True)
    print(model.predict(np.stack(hdf.histDf.tail(1).values[0])))
