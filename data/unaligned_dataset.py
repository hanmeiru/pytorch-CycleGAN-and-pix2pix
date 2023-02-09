import os
from data.base_dataset import BaseDataset, get_transform
from data.npz_folder import make_dataset
from PIL import Image
import random
from scipy.sparse import load_npz
import numpy as np 
import os 
import matplotlib.pyplot as plt
from matplotlib import cm
import io

def remove_unpaired(dirA, dirB):
    file_root_A = set([i[:-5] for i in os.listdir(dirA)]) # remove extension and "A" or "B"
    file_root_B = set([i[:-5] for i in os.listdir(dirB)])
    common_file_root = file_root_A.intersection(file_root_B)
    for f in file_root_A:
        if f not in common_file_root:
            os.remove(dirA + "/" + f + 'A.npz')
            print(f, 'A.npz removed')
    for f in file_root_B:
        if f not in common_file_root:
            os.remove(dirB + "/" + f + 'B.npz')
            print(f, 'B.npz removed')

class UnalignedDataset(BaseDataset):
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
        remove_unpaired(self.dir_A, self.dir_B) # remove unpaired images 

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB' 

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print(self.A_size)
        print(self.B_size)
        # split training and testing datasets
        # train_size = int(self.A_size * 0.7) 
        assert self.A_size == self.B_size #

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within range
        B_path = A_path.replace("trainA", "trainB")[:-5]+'B.npz' # A.npz->B.npz
       
        # convert A from sparse matrix to ndarray to img
        sparse_A = load_npz(A_path) # load sparse matrix from path
        array_A = np.array(sparse_A.toarray()) #convert into array
        array_A[:,0]=array_A[:,0]/10 # first column
        A_img = Image.fromarray(255 - (array_A * 255 / np.max(array_A)).astype('uint8'), mode = "L")
        
        # convert B from sparse matrix to ndarray to img
        sparse_B = load_npz(B_path) # load sparse matrix from path
        array_B = np.array(sparse_B.toarray(), dtype = np.uint8) #convert into array
        array_B[:,0]=array_B[:,0]/10 # first column
        B_img = Image.fromarray(255 - (array_B * 255 / np.max(array_B)).astype('uint8'), mode = "L")

        ### original
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')


        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

