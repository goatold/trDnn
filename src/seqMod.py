#!/bin/env python3
import glob
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint

import dataPrep
import conf

if __name__ == '__main__':
    ds = dataPrep.readDataFromFile(glob.glob(conf.data_files), conf.COL_NAMES, conf.EXCEL_COL_TO_READ)
    train_data, train_label = ds.getDataSets(-conf.sampleSizeT, conf.trainDataBefore)
    valid_data, valid_label = ds.getDataSets(conf.sampleSizeV, conf.validDataAfter)
    # consider keras.utils.to_categorical(label, num_classes=NUM_CLASS)

    # create sequential model with m LSTM layers and n Dense layers
    # we may replace LSTM with CuDNNLSTM for GPU
    model = Sequential()
    for i in range(conf.NUM_OF_LSTM):
        model.add(LSTM(conf.lstmOutSize, input_shape=(train_data.shape[1:]), return_sequences=(i<conf.NUM_OF_LSTM-1)))
        # helps prevent overfitting
        model.add(Dropout(rate=conf.dropRatio))
        model.add(BatchNormalization())

    for i in range(conf.NUM_OF_DENSE-1):
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(rate=conf.dropRatio))

    model.add(Dense(conf.NUM_CLASS, activation='softmax'))


    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(conf.LOG_NAME))

    # unique file name that will include the epoch and the validation acc for that epoch
    filepath = "trdnn-{epoch:02d}-{val_acc:.3f}"
    # saves only the best ones
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Train model
    history = model.fit(
        train_data, train_label,
        batch_size=conf.BATCH_SIZE,
        epochs=conf.EPOCHS,
        validation_data=(valid_data, valid_label),
        callbacks=[tensorboard, checkpoint],
    )

    # Score model
    score = model.evaluate(valid_data, valid_label, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(conf.LOG_NAME))
    # we may later load saved model to resume training or predict new data
    # model = load_model('path_to_model')

