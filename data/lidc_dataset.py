"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
class LIDCDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        # parser.set_defaults(preprocess_mode='resize_and_crop')
        # load_size = 286 if is_train else 256
        # parser.set_defaults(load_size=load_size)
        # parser.set_defaults(crop_size=256)
        # parser.set_defaults(display_winsize=256)
        # parser.set_defaults(label_nc=2)
        # parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, default = '/home/bme001/20180883/data/LIDC_Data/train_label', required=False,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str,  default = '/home/bme001/20180883/data/LIDC_Data/train_image', required=False,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = []
        label_list = sorted(os.listdir(label_dir))
        for label in label_list:
            label_paths.append(os.path.join(label_dir, label))
        image_dir = opt.image_dir
        image_paths = []
        image_list = sorted(os.listdir(image_dir))
        for image in image_list:
            image_paths.append(os.path.join(image_dir, image))


        instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
    
    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        
        instance_tensor = 0


        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
