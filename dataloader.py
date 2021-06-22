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

class DataSet:

    images = []
    labels = []
    def __init__(self, 
                data_dir=None,
                csv_file='image_set.dat', 
                header_file='header.tfl.txt',
                transform = None,
                file_ending = ".png",
                num_classes = 10,
                train = False,
                img_dim = 32):
    
        self.data_dir = data_dir
        self.header_file = header_file
        self.csv_file = csv_file
        self.transform = transform
        self.dataset = None
        self.classlist = None
        self.train = train
        self.file_ending = file_ending

        self.load_data()
        self.images, self.labels = self.load_images_and_labels(img_dim)
        self.save_data(dir=data_dir)

    def __repr__(self):
        return f"Number of datapoints: {str(len(self.dataset))}, \n Root location: {self.data_dir}, Transform: {self.transform} \nTrain: {self.train}"

    def __str__(self):
        return f"Number of datapoints: {str(len(self.dataset))}, \n Root location: {self.data_dir}, \n Transform: {self.transform} \nTrain: {self.train}"

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
        elif self.file_ending == ".png":
            image = io.imread(img_name, pilmode='RGBA')
        else:
            image = io.imread(img_name)

        label = self.dataset.iloc[idx, 0].split(' ')[1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
    def load_images_and_labels(self, img_dim):
        
        images = []
        labels = []
        DIM = img_dim

        try:
            for idx in range(len(self.dataset)):

                img_name = self.dataset.iloc[idx, 0].split(' ')[0]
                label = self.dataset.iloc[idx, 0].split(' ')[1]

                img = Image.open(img_name)
                img = img.convert('RGB')
         
                if idx % 1000 == 0:
                    print(idx)

                img = np.asarray(img)
                img = cv2.resize(img, dsize=(DIM, DIM), interpolation=cv2.INTER_CUBIC)
                img = img.transpose(2,0,1)
     
                images.append(img)
                labels.append(label)
                            
            images = np.asarray(images)
            labels = np.asarray(labels, dtype="int64")
            images = images.reshape(-1, 3, DIM, DIM)
            images = images.transpose(0, 2, 3, 1)  # convert to HWC


        except Exception as e:
            print(str(e))

        return images, labels


    def split_into_train_test(self):
        pass

    def save_data(self, dir):
        
        np.save(f'{dir}/data.npy', self.images)    
        np.save(f'{dir}/labels.npy', self.labels)    

  

    def load_data(self):
        if os.path.isfile(os.path.join(self.data_dir, self.header_file)):
            self.get_classes_from_file()
        else:
            self.get_classes_from_directory()
            self.save_classes_to_file()
        if os.path.isfile(os.path.join(self.data_dir, self.csv_file)):
            self.get_data_from_file()
        else:
            self.get_data_from_directory()
            self.save_data_to_file()
        self.dataset = pd.read_csv(os.path.join(self.data_dir, self.csv_file))


    def convertToNumpy(self):

        for idx in range(len(self.dataset)):

            img_name = self.dataset.iloc[idx, 0].split(' ')[0]
            if self.file_ending == ".png":
                image = io.imread(img_name, pilmode='RGB')
                image = np.array(image)
            elif self.file_ending == ".jpg":
                image = io.imread(img_name)
                image = np.array(image)
            label = self.dataset.iloc[idx, 0].split(' ')[1]
            self.images.append(image)
            self.labels.append(int(label))
        
        self.images = np.array(self.images, dtype=object)
        self.labels = np.array(self.labels, dtype=object)
        sample = {'images': self.images, 'labels': self.labels}
        
        return self.images, self.labels
    
    def get_data(self):
        '''
        Read the data file and get the list of images along with their labels
        :return the data set
        '''
        input_data = pd.read_csv(os.path.join(self.data_dir, self.csv_file), header=None, delimiter=' ')
        print(input_data.head())
        return input_data

    def get_classes(self):
        '''
        Get the list of classes from the header file
        return: clst  the class list
        '''
        cl_file = self.header_file
        with open(cl_file) as f:
            reader = csv.reader(f)
            cl = [r for r in reader]
        clst = cl[0]
        return clst


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
    

class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype(np.uint8))
            #x = Image.fromarray((x * 255).astype(np.uint8)) # Added to fix some bug, consider removing

            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)



if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
