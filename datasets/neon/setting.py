from easydict import EasyDict as edict
import os

# init
__C_OP = edict()

cfg_data = __C_OP

__C_OP.STD_SIZE = (400,400)
__C_OP.TRAIN_SIZE = (400,400) # 2D tuple or 1D scalar
# __C_OP.DATA_PATH = '../ProcessedData/Shanghai_proA'
__C_OP.DATA_PATH = '/content/gdrive/MyDrive/Documents/Masters/7CCSMPRJ - Individual Project/experiments/data/neon'
# __C_OP.UPSAMPLE = True
__C_OP.MEAN_STD = (
    [0.5928, 0.5642, 0.4799],
    [0.1659, 0.1541, 0.1125]
)

__C_OP.LABEL_FACTOR = 1
__C_OP.LOG_PARA = 100.

__C_OP.RESUME_MODEL = ''#model path
__C_OP.TRAIN_BATCH_SIZE = 1 #imgs

__C_OP.VAL_BATCH_SIZE = 1 # must be 1


