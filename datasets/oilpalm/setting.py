from easydict import EasyDict as edict
import os

# init
__C_OP = edict()

cfg_data = __C_OP

__C_OP.STD_SIZE = (768,1024)
__C_OP.TRAIN_SIZE = (576,768) # 2D tuple or 1D scalar
# __C_OP.DATA_PATH = '../ProcessedData/Shanghai_proA'
__C_OP.DATA_PATH = '/content/gdrive/MyDrive/Documents/Masters/7CCSMPRJ - Individual Project/experiments/data/oilpalm'
# __C_OP.UPSAMPLE = True
__C_OP.MEAN_STD = (
    [0.4053, 0.4637, 0.3644],
    [0.1531, 0.1227, 0.1198]
)

__C_OP.LABEL_FACTOR = 1
__C_OP.LOG_PARA = 100.

__C_OP.RESUME_MODEL = ''#model path
__C_OP.TRAIN_BATCH_SIZE = 1 #imgs

__C_OP.VAL_BATCH_SIZE = 1 # must be 1


