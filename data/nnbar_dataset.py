import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset_ub
#from data.image_folder import make_dataset
from util.config import load_train_image_filenames
from util.loaddata import load_npz
from PIL import Image
import random
import torch

import numpy as np

class NnbarDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA' "/a/data/bleeker/yjwa/datasets/trainA/"
    and from domain B '/path/to/data/trainB' respectively. "/a/data/bleeker/yjwa/datasets/trainB/"
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    CONFIG_A_YML='config/config.nnbar_trainA.yml'
    CONFIG_B_YML='config/config.nnbar_trainB.yml'

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA' dtaopt.phase="train"
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        CONFIG_A_YML='/data/yjwa/torchwork/pytorch-CycleGAN-and-pix2pix/data/config/config.nnbar_trainA.yml'
        CONFIG_B_YML='/data/yjwa/torchwork/pytorch-CycleGAN-and-pix2pix/data/config/config.nnbar_trainB.yml'

        self.A_train_files = load_train_image_filenames(CONFIG_A_YML)
        self.B_train_files = load_train_image_filenames(CONFIG_B_YML)
                    
        self.A_train = np.array([], dtype=np.float).reshape(0, 3, 400, 400)
        self.B_train = np.array([], dtype=np.float).reshape(0, 3, 400, 400)
                               
        for A_train_file in self.A_train_files:
            print("INFO: load data:", A_train_file)
            self.A_train = np.concatenate((self.A_train, load_npz(A_train_file)))
        for B_train_file in self.B_train_files:
            print("INFO: load data:", B_train_file)
            self.B_train = np.concatenate((self.B_train, load_npz(B_train_file)))

        self.A_size = len(self.A_train)  # get the size of dataset A
        self.B_size = len(self.B_train)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        #self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        #self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        #A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_path = self.A_train_files[0]
        B_path = self.B_train_files[0]#not reaally needed

        A_entry = self.A_train[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        #B_path = self.B_paths[index_B]
        B_entry = self.B_train[index_B]

#        A_img = Image.open(A_path).convert('RGB') # yj A_path is */*.jpg...        
#        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        #A = self.transform_A(A_img)
        #B = self.transform_B(B_img)


        A = A_entry
        B = B_entry
        #A = A.torch.cuda.FloatTensor()
        #A = A.to(torch.float32)
        #print(A.dtype)
        #print(A_path)
#        B = B.torch.cuda.FloatTensor()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
