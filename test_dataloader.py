

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

# parse options
opt = TrainOptions().parse()
opt.dataset_mode = 'lidc'
opt.label_dir = '/home/bme001/20180883/data/LIDC_Data/train_label'

opt.image_dir = '/home/bme001/20180883/data/LIDC_Data/train_img'
opt.no_instance = True
opt.label_nc = 2
opt.semantic_nc = 2
opt.contain_dontcare_label = False
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

for i, data_i in enumerate(dataloader):
    if i > 2:
        break

    print('shape of the image is {} and shape of the label is {}'.format(data_i['image'].shape,data_i['label'].shape) )