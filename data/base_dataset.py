"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

import cv2
cv2.setNumThreads(0)        # avoid deadlock w/ multiprocess dataloader
cv2.ocl.setUseOpenCL(False) # avoid deadlock w/ multiprocess dataloader

import albumentations as ab   # data aug
import albumentations.pytorch # data aug
from functools import partial

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def transform(image, mask=None, opt=None, paired=False):
    assert opt is not None, 'opt is not optional (it cannot be None)'
    #[ ab.Resize(height=opt.load_height, width=opt.load_width)] if opt.load_width!=-1 else [])
    resize = ab.Resize(height=opt.load_height, width=opt.load_width)
    T_stack_pre = ab.Compose( 
          [ 
            #ab.ChannelDropout((1, 4 if paired else 2 )),
            #ab.ChannelDropout(),
            ab.OpticalDistortion(),
            ab.RandomCrop(height=opt.crop_size, width=opt.crop_size),
            # ab.CoarseDropout()
        ] )
    T_pair = ab.Compose(  [
            ab.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.01, rotate_limit=5, interpolation=1),
            ab.OneOf([
                ab.MotionBlur(),
                ab.MedianBlur(5),
                ab.GaussianBlur(),
            ]),
            #ab.ToFloat(),
            #ab.HueSaturationValue(),
            #ab.RandomBrightness(),
            #ab.RandomContrast(),
            ab.Normalize((.5,)*3, (.5,)*3)
        ])
    #nc = 6 if paired else 3
    T_stack_post = ab.Compose([
            #ab.Normalize((0.5,)*nc, (0.5,)*nc),
            ab.pytorch.ToTensor(),
        ])



    if paired:
        #i = 0
        #print('\n\n')
        #print(str(i), image.mean(), image.std()); i += 1

        aug = resize(image=image, mask=mask)
        image, mask = aug['image'], aug['mask']

        #print(str(i), image.mean(), image.std()); i += 1
        cat = T_stack_pre(image=np.concatenate((image, mask), axis=-1))['image']
        image, mask = cat[..., :3], cat[..., 3:]

        #print(str(i), image.mean(), image.std()); i += 1
        aug = T_pair(image=image, mask=mask)
        image, mask = aug['image'], aug['mask']

        #print(str(i), image.mean(), image.std()); i += 1
        cat = T_stack_post(image=np.concatenate((image, mask), axis=-1))['image']
        image, mask = cat[:3], cat[3:]
        #print(str(i), image.mean(), image.std()); i += 1
        #print('\n\n')
        return image, mask
    else:
        #print('IMG', type(image)) 
        #print('1',image.shape, end=' -> ',flush=True)
        image = resize(image=image)['image'],
        #print('\n\n\n\n\n\n\n')
        #print(image)
        #print('\n\n\n\n\n\n\n')
        #print('2',image.shape, end=' -> ', flush=True)
        #print('IMG TYPE',type(image), len(image))
        if type(image) is tuple:
            image = image[0]
        image = T_stack_pre(image=image)['image']
        #print(image.shape, end=' -> ',flush=True)

        image = T_pair(image=image)['image']
        #print(image.shape, end=' -> ')

        image = T_stack_post(image=image)['image']
        #print(image.shape)

        return image



def get_transform(
    opt, 
    params=None, 
    grayscale=False, 
    method=Image.BICUBIC, 
    convert=True, 
    paired=False ):

    return partial(transform,opt=opt,paired=paired)
    # transform_list = []
    # if paired:
    #     # new_h = new_w = None
    #     # if opt.preprocess == 'resize_and_crop':
    #     #     new_h = new_w = opt.load_size
    #     # elif opt.preprocess == 'scale_width_and_crop':
    #     #     new_w = opt.load_size
    #     #     new_h = opt.load_size * h // w
    #     new_w, new_h = opt.load_width, opt.load_height


    #     T = ab.Compose([ab.Resize(height=new_h, width=new_w) for new_w, new_h in ((new_w, new_h),) if new_w!=-1] + [
    #         ab.RandomCrop(height=opt.crop_size, width=opt.crop_size),
    #      #   ab.RandomShadow(),
    #      #   ab.RandomSunFlare(p=.1),
    #        # ab.OpticalDistortion(),
    #        # ab.HueSaturationValue(),
    #        # ab.RandomBrightness(),
    #        # ab.RandomContrast(),
    #         # ab.OneOf([
    #             # ab.MotionBlur(),
    #             # ab.MedianBlur(),
    #             # ab.GaussianBlur(),
    #         # ]),
    #        # ab.CoarseDropout(),
    #         ab.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     #    ab.ChannelDropout(),
    #         ab.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.1, rotate_limit=10, interpolation=1),
    #         ab.pytorch.ToTensor(sigmoid=False)
    #     ])

    #     return T


    #transform_list = []
    #if grayscale:
    #    transform_list.append(transforms.Grayscale(1))
    #if 'resize' in opt.preprocess:
    #    osize = [opt.load_size, opt.load_size] if opt.load_width==-1 else [opt.load_height,opt.load_width]
    #    transform_list.append(transforms.Resize(osize, method))
    #elif 'scale_width' in opt.preprocess:
    #    transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size if opt.load_width==-1 else opt.load_width, method)))
#
    #if 'crop' in opt.preprocess:
    #    if params is None:
    #        transform_list.append(transforms.RandomCrop(opt.crop_size))
    #    else:
    #        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
#
    #if opt.preprocess == 'none':
    #    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
#
    #if not opt.no_flip:
    #    if params is None:
    #        transform_list.append(transforms.RandomHorizontalFlip())
    #    elif params['flip']:
    #        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
#
    #if convert:
    #    transform_list += [transforms.ToTensor()]
    #    if grayscale:
    #        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    #    else:
    #        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    #return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
