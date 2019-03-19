#!/bin/env python3
import glob
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time

import dataPrep

EPOCHS = 16  # how many passes through our data
BATCH_SIZE = 8  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
LOG_NAME = f"{dataPrep.RETRO_LEN}-RETRO-{dataPrep.CLASS_PCT}-CLP-{int(time.time())}"

if __name__ == '__main__':
    sampleSizeT, sampleSizeV = 712, 40
    data_files = 'data/*.xlsx'
    ds = dataPrep.readDataFromFile(glob.glob(data_files))
    train_data, train_label, valid_data, valid_label = ds.getDataSets(sampleSizeT, sampleSizeV)
    # consider keras.utils.to_categorical(label, num_classes=NUM_CLASS)

    # create sequential model with 3 LSTM layers and 2 Dense layers
    # we may replace LSTM with CuDNNLSTM for GPU
    dropRatio = 0.02
    lstmOutSize = 128
    model = Sequential()
    model.add(LSTM(lstmOutSize, input_shape=(train_data.shape[1:]), return_sequences=True))
    # helps prevent overfitting
    model.add(Dropout(rate=dropRatio))
    model.add(BatchNormalization())

    model.add(LSTM(lstmOutSize, return_sequences=True))
    model.add(Dropout(dropRatio))
    model.add(BatchNormalization())

    model.add(LSTM(lstmOutSize))
    model.add(Dropout(dropRatio))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropRatio))

    model.add(Dense(dataPrep.NUM_CLASS, activation='softmax'))


    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(LOG_NAME))

    # unique file name that will include the epoch and the validation acc for that epoch
    filepath = "trdnn-{epoch:02d}-{val_acc:.3f}"
    # saves only the best ones
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Train model
    history = model.fit(
        train_data, train_label,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(valid_data, valid_label),
        callbacks=[tensorboard, checkpoint],
    )

    # Score model
    score = model.evaluate(valid_data, valid_label, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(LOG_NAME))
    # we may later load saved model to resume training or predict new data
    # model = load_model('path_to_model')

