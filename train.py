import os
import numpy as np
import torch
import argparse
from datetime import datetime
from config import cfg

parser = argparse.ArgumentParser(description='c3')

parser.add_argument('--model', '-m', metavar='MODEL', type=str,
                    help='model name')
parser.add_argument('--sigma', '-s', metavar='MODEL', default=4, type=int,
                    help='sigma size')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')
parser.add_argument('--maxepoch', '-mep', metavar='MAXEPOCH', default=20,type=int,
                    help='maximum number of epochs')
parser.add_argument('--lr', '-l', metavar='LEARNINGRATE', default=1e-6,type=float,
                    help='maximum number of epochs')
parser.add_argument('--data', '-d', metavar='DATA', default=None, type=str,
                    help='maximum number of epochs')

args = parser.parse_args()
if args.lr:
    cfg.LR = args.lr
if args.data:
    cfg.DATASET = args.data
if args.maxepoch:
    cfg.MAX_EPOCH = args.maxepoch
if args.sigma:
    cfg.SIGMA = args.sigma
if args.model:
    cfg.NET = args.model
    now = datetime.today().strftime("%m-%d_%H-%M")
    cfg.EXP_NAME = f'{now}_{cfg.DATASET}_{cfg.NET}_{cfg.LR}_{cfg.SIGMA}'
if args.pre:
    cfg.RESUME = True
    cfg.RESUME_PATH = os.path.join(
        os.getcwd(),
        'exp',
        args.pre,
        'latest_state.pth'
    )

print(cfg.EXP_NAME)


#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = cfg.DATASET
print(data_mode)
if data_mode is 'SHHA':
    from datasets.SHHA.loading_data import loading_data 
    from datasets.SHHA.setting import cfg_data 
elif data_mode is 'SHHB':
    from datasets.SHHB.loading_data import loading_data 
    from datasets.SHHB.setting import cfg_data 
elif data_mode is 'QNRF':
    from datasets.QNRF.loading_data import loading_data 
    from datasets.QNRF.setting import cfg_data 
elif data_mode is 'UCF50':
    from datasets.UCF50.loading_data import loading_data 
    from datasets.UCF50.setting import cfg_data 
elif data_mode is 'WE':
    from datasets.WE.loading_data import loading_data 
    from datasets.WE.setting import cfg_data 
elif data_mode is 'GCC':
    from datasets.GCC.loading_data import loading_data
    from datasets.GCC.setting import cfg_data
elif data_mode is 'Mall':
    from datasets.Mall.loading_data import loading_data
    from datasets.Mall.setting import cfg_data
elif data_mode is 'UCSD':
    from datasets.UCSD.loading_data import loading_data
    from datasets.UCSD.setting import cfg_data
elif data_mode == 'oilpalm':
    from datasets.oilpalm.loading_data import loading_data
    from datasets.oilpalm.setting import cfg_data
elif data_mode == 'blan':
    from datasets.blan.loading_data import loading_data
    from datasets.blan.setting import cfg_data
elif data_mode == 'neon':
    from datasets.neon.loading_data import loading_data
    from datasets.neon.setting import cfg_data
elif data_mode == 'london':
    from datasets.london.loading_data import loading_data
    from datasets.london.setting import cfg_data

#------------Prepare Trainer------------
net = cfg.NET
print(net)
print(datetime.today())
# if net in ['MCNN', 'AlexNet', 'VGG', 'VGG_DECODER', 'Res50', 'Res101', 'CSRNet','Res101_SFCN',
#            'arch11','arch12','arch13','arch14','arch15','arch21','arch22','arch23','arch24',
#            'SASNet', 'UNet']:

if 'SANet' in net:
    from trainer_for_M2TCC import Trainer # double losses but signle output
else:
    from trainer import Trainer

# elif net in ['CMTL']:
#     from trainer_for_CMTL import Trainer # double losses and double outputs
# elif net in ['PCCNet']:
#     from trainer_for_M3T3OCC import Trainer

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data,cfg_data,pwd)
cc_trainer.forward()
