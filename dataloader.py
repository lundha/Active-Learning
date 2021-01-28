

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset
from torchvision import transforms, utils
import imageio
import csv
import cv2
import random
from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split


class PlanktonDataSet(Dataset):

    dataset = None
    classlist = None

    def __init__(self, 
                data_dir='/Users/martin.lund.haug/Documents/Prosjektoppgave/Datasets/datasciencebowl/train',
                csv_file='image_set.dat', 
                header_file='header.tfl.txt',
                transform = None,
                file_ending = ".jpg",
                num_classes = 121):
    
        self.data_dir = data_dir
        self.header_file = header_file
        self.csv_file = csv_file
        self.transform = transform
        
        self.file_ending = file_ending
        self._classes = num_classes
        self.load_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.dataset.iloc[idx, 0].split(' ')[0]
        if self.file_ending == ".tiff":
            image = Image.open(img_name)
            image = image.convert('RGB')
            image = np.array(image)
        else:
            image = io.imread(img_name)
        label = self.dataset.iloc[idx, 0].split(' ')[1]
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


    def load_data(self):
        if os.path.isfile(os.path.join(self.data_dir, self.header_file)):
            print('path ', os.path.join(self.data_dir, self.header_file))
            self.get_classes_from_file()
        else:
            self.get_classes_from_directory()
            self.save_classes_to_file()
        if os.path.isfile(os.path.join(self.data_dir, self.csv_file)):
            print('path ', os.path.join(self.data_dir, self.header_file))
            self.get_data_from_file()
        else:
            self.get_data_from_directory()
            self.save_data_to_file()
        self.dataset = pd.read_csv(os.path.join(self.data_dir, self.csv_file))


    def get_classes_from_file(self):

        print('Get the list of classes from the header file ', self.header_file)

        with open(self.header_file) as f:
            reader = csv.reader(f)
            cl = [r for r in reader]
        self.classlist = cl[0]
        return self.classlist

    def get_data_from_file(self):

        '''
        Read the data file and get the list of images along with their labels
        and assign the input_data to the data set
        '''
        print('Get data from file ', os.path.join(self.data_dir, self.csv_file))
        self.dataset = pd.read_csv(os.path.join(self.data_dir, self.csv_file), header=None, delimiter=' ')
        #print(self.dataset.head())


    def get_classes_from_directory(self):
        print("Get classes from the database directory: ", self.data_dir)
        self.classlist = [_class for _class in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, _class))]
        print("List of classes from the directory ", self.classlist)

    def get_data_from_directory(self):

        fileList = []
        for class_idx, _class in enumerate(self.classlist):
            print(" ", _class)
            filepath = os.path.join(self.data_dir, _class) # Path to data files
            files = [_file for _file in os.listdir(filepath) if _file.endswith(self.file_ending)]
            for f in files:
                fileList.append([os.path.join(filepath, f), str(class_idx)])
        fileList = np.array(fileList)
        print("Shuffle dataset...")
        np.random.shuffle(fileList)
        self.dataset = fileList
        print("*** Dataset ***")
        print(self.dataset)

    
    def get_classes_from_plankton_directory(self):

        for cat in range(0, self._classes):
            cat_dir = glob.glob(
            os.path.join(self.data_dir, '%03d*' % (cat + 1)))[0]
        for img_file in glob.glob(os.path.join(cat_dir, '*.jpg')):        
            self.classlist.append(cat)


    def get_data_from_plankton_directory(self):
        
        for cat in range(0, self._classes):
            cat_dir = glob.glob(
            os.path.join(self.root_dir, '%03d*' % (cat + 1)))[0]
        for img_file in glob.glob(os.path.join(cat_dir, '*.jpg')):
            self.dataset.append(img_file)
    

    def save_classes_to_file(self):

        # First save classes to a CSV file
        print('Save classes to file ', self.header_file)
        df_classes = pd.DataFrame(columns=self.classlist)
        df_classes.to_csv(self.header_file, index=False)

    def save_data_to_file(self):

        # Secondly save data 
        print('Save into the data file ....', self.csv_file)
        np.savetxt(self.csv_file, self.dataset, delimiter=' ', fmt='%s')

class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': label}

class Convert2RGB(object):
    """Convert image to rgb if in gray scale.
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Converting image to rgb if it is in grayscale
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        sample = {'image': image, 'label': label}    
        return sample


class Normalize(object):
  
    def __init__(self, mean: np.ndarray = np.array([0.5, 0.5, 0.5]),  
                    std: np.ndarray = np.array([0.5, 0.5, 0.5])): 
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        img, label = sample['image'], sample['label']
        img = img - self.mean
        img /= self.std
        sample = {'image': img, 'label': label}
        return sample

  
class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
    
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(np.array([int(label)]))}
    


if __name__ == "__main__":

    data_dir = "/Users/martin.lund.haug/Documents/Prosjektoppgave/Datasets/plankton_new_data/Dataset_BeringSea/train/"
    header_file = data_dir + "header.tfl.txt"
    filename = data_dir + "image_set.data"

    composed = transforms.Compose([Convert2RGB(), Resize(64), ToTensor()])
    
    try:
        dataset = PlanktonDataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=".bmp",
                                    transform=composed)
    except Exception as e:
        print("Could not load dataset, error msg: ", str(e))

    uncert_samp_idx = [1,2,3,403,493,24,3454,23,354,2365,877,3436,879,9896,232,5,9]
    classCount  = [0]*10

    print("*******")
    # Get classes for the uncertainty samples
    for idx in uncert_samp_idx:
        label = dataset.dataset.iloc[idx, 0].split(' ')[1]
        classCount[int(label)] += 1
    print(classCount)