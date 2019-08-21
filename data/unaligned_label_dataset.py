import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

import torch
import numpy as np
import skimage # to read tiffs
import skimage.io # to read tiffs

class UnalignedLabelDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        ### added by jack ### 
        verbose = self.opt.verbose
        print("verbose:", verbose)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        print(''.join((opt.phase, 'Lbl', 'A')))
        self.lbl_dir_A = os.path.join(opt.dataroot, ''.join((opt.phase, 'Lbl', 'A')))  # create a path '/path/to/data/trainLblA'
        self.lbl_dir_B = os.path.join(opt.dataroot, ''.join((opt.phase, 'Lbl', 'B')))  # create a path '/path/to/data/trainLblB'

        self.hasLblA = os.path.exists(self.lbl_dir_A)
        self.hasLblB = os.path.exists(self.lbl_dir_B)

        print('A has label: {}\t B has label: {}'.format(self.hasLblA, self.hasLblB))

        if self.hasLblA:
            if verbose:
                print("gathering A label", end="\t", flush=True)
            self.A_lbl_paths = sorted(make_dataset(self.lbl_dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainLblA'
            if verbose:
                print("done -- {} images".format(len(self.A_lbl_paths)), flush=True)

            self.paired_transformA = get_transform(self.opt, paired=True)
        else:
            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))

        if self.hasLblB:
            if verbose:
                print("gathering B label", end="\t", flush=True)
            self.B_lbl_paths = sorted(make_dataset(self.lbl_dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainLblB'
            if verbose:
                print("done -- {} images".format(len(self.B_lbl_paths)), flush=True)

            self.paired_transformB = get_transform(self.opt, paired=True)
        else:
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        ### /added by jack ### 

        if verbose:
            print("gathering A", end="\t", flush=True)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        if verbose:
            print("done -- {} images".format(len(self.A_paths)), flush=True)
        if verbose:
            print("gathering A", end="\t", flush=True)
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        if verbose:
            print("done -- {} images".format(len(self.B_paths)), flush=True)

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size # make sure index is within range
        A_path = self.A_paths[index_A]  
        if self.opt.serial_batches:   # make sure index is within range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)


        ### added by jack -- get labels and concatenate with inputs ### 
        A_lbl_path = B_lbl_path = None
        if self.hasLblA:
            A_lbl_path = self.A_lbl_paths[index_A]  # use index from above
            A_lbl = np.asarray(skimage.io.imread(A_lbl_path))
            A_img = np.asarray(A_img)
           # print("A: ",A_img.shape, A_lbl.shape)
            aug = self.paired_transformA(image=A_img, mask=A_lbl)
            img, lbl = aug['image'], np.transpose(aug['mask'][0], (2,0,1))
           # print("A_a: ",img.shape, lbl.shape)
            A = torch.cat((img, lbl))
        else:
            A = self.transform_A(A_img)

        if self.hasLblB:
            B_lbl_path = self.B_lbl_paths[index_B] # use index from above
            B_lbl = np.asarray(skimage.io.imread(B_lbl_path))
            B_img = np.asarray(B_img)
           # print("B:", B_img.shape, B_lbl.shape)
            aug = self.paired_transformB(image=B_img, mask=B_lbl)
            img, lbl = aug['image'], np.transpose(aug['mask'][0], (2,0,1))
           # print("B_a: ",img.shape, lbl.shape)
            B = torch.cat((img, lbl))
        else:
           # print(B_img)
           # print(type(B_img))
           # print(B_img.size, np.asarray(B_img).shape)
           # print()
            B = self.transform_B(B_img)

        # pad images with less channels 
        #if False and self.hasLblA != self.hasLblB:
        #    s, l = (A, B) if A.shape[0] < B.shape[0] else (B, A)
        #    diff = len(l)-len(s)
        #    padding = torch.zeros((diff, *s.shape[1:]))
           # print("s l pad: ", s.shape, l.shape, padding.shape, end=' ')
        #    s = torch.cat((s, padding))
           # print(s.shape)
        #    if A.shape[0] < B.shape[0]:
        #        A, B = s, l 
        #    else:
        #        B, A = s, l
        ### /added by jack ### 

        return {
                'A': A, 'B': B, 
                'A_paths': A_path, 'B_paths': B_path,
                'A_lbl_paths': A_lbl_path if A_lbl_path is not None else '',
                'B_lbl_paths': B_lbl_path if B_lbl_path is not None else ''
            }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
