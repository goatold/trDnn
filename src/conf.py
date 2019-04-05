import time

# data preprocess conf
RETRO_LEN = 20
CLASS_PCT = 0.01
NUM_CLASS = 3
VALIDATION_SET_PCT = 0.05
PREDIC_DAYS = 1

TABLE_NAMES = ('hs300', 'wind_index_A')
TARGET_TABLE = TABLE_NAMES[0]
COL_NAMES = ('time', 'open', 'high', 'low', 'close',)
TARGET_COL = COL_NAMES[-1]
EXCEL_COL_TO_READ = 'c:g'

data_files = 'data/*.xlsx'

# model training conf
EPOCHS = 16  # how many passes through our data
BATCH_SIZE = 8  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
LOG_NAME = f"{RETRO_LEN}-RETRO-{CLASS_PCT}-CLP-{time.strftime('%Y%m%d_%H%M%S',time.gmtime())}"

trainDataBefore = '2018-09-30'
validDataAfter = '2018-10-01'
sampleSizeT, sampleSizeV = 3200, 64

dropRatio = 0.02
lstmOutSize = 128
