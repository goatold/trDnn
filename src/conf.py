import time

# data preprocess conf
RETRO_LEN = 20
CLASS_PCT = 0.01
NUM_CLASS = 3
VALIDATION_SET_PCT = 0.05
PREDIC_DAYS = 1

TABLE_NAMES = ('hs300', 'wind_index_A')
TARGET_TABLE = TABLE_NAMES[0]
COL_NAMES = ('time', 'open', 'high', 'low', 'close', 'volume')
TARGET_COL = COL_NAMES[-2]
EXCEL_COL_TO_READ = 'c:g,i'

data_files = 'data/*.xlsx'

# model training conf
EPOCHS = 16  # how many passes through our data
BATCH_SIZE = 16  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
LOG_NAME = f"{RETRO_LEN}-RETRO-{CLASS_PCT}-CLP-{time.strftime('%Y%m%d_%H%M%S',time.gmtime())}"
NUM_OF_LSTM = 4
NUM_OF_DENSE = 2

trainDataBefore = '2018-07-11'
validDataAfter = '2018-07-12'
sampleSizeT, sampleSizeV = 3264, 128

dropRatio = 0.05
lstmOutSize = 128
