import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035 # random seed,  for reproduction
__C.DATASET = 'neon' # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD

if __C.DATASET == 'UCF50':# only for UCF50
	from datasets.UCF50.setting import cfg_data
	__C.VAL_INDEX = cfg_data.VAL_INDEX 

if __C.DATASET == 'GCC':# only for GCC
	from datasets.GCC.setting import cfg_data
	__C.VAL_MODE = cfg_data.VAL_MODE



__C.NET = 'CSRNet' # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet

__C.PRE_GCC = False # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model' # path to model

__C.RESUME = False # contine training
# __C.RESUME_PATH = './exp/04-25_09-19_SHHB_VGG_1e-05/latest_state.pth' #
__C.RESUME_PATH = os.path.join(
	os.getcwd(),
	'exp',
	# '07-11_16-24_oilpalm_CSRNet_1e-05',
	# '07-12_17-49_oilpalm_CSRNet_1e-06',
	'latest_state.pth'
)

__C.GPU_ID = [0] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-6 # learning rate
__C.LR_DECAY = 0.995 # decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 50
__C.SIGMA = 4

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on
__C.LAMBDA_1 = 1e-3# SANet:0.001 CMTL 0.0001
print(f'second learning rate {__C.LAMBDA_1}')

# print 
__C.PRINT_FREQ = 40

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = f'{now}_{__C.DATASET}_{__C.NET}_{__C.LR}_{__C.SIGMA}'

if __C.DATASET == 'UCF50':
	__C.EXP_NAME += '_' + str(__C.VAL_INDEX)	

if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_' + __C.VAL_MODE	

__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 1 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes



#================================================================================
#================================================================================
#================================================================================  
