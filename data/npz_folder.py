"""A modified npz folder class

We modify the modified PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load npz files from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from scipy.sparse import load_npz


def is_npz_file(filename):
    return filename.endswith(".npz")


def make_dataset(dir, max_dataset_size=float("inf")):
    npz_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_npz_file(fname):
                path = os.path.join(root, fname)
                npz_files.append(path)
    return npz_files[:min(max_dataset_size, len(npz_files))]


def default_loader(path):
    sparse_matrix = load_npz(path) # load sparse matrix from path
    arr = np.array(sparse_matrix.toarray(), dtype = np.uint8) #convert into array
    img = Image.fromarray(arr * 255, mode = "L") # mode = 'RGB'
    return img


class npzFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        npz = make_dataset(root)
        if len(npz) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported extension is " + ".npz"))

        self.root = root
        self.imgs = npz
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        npz = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return npz, path
        else:
            return npz

    def __len__(self):
        return len(self.npz)
