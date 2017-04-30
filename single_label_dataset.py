"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
from PIL import Image
import os
import numpy as np
import sys


class Dataset:
    def __init__(self, train_list, valid_list, test_list, database_root, data_aug=False):
        """Initialize the Dataset object, scale image to 224x224
        Args:
        train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        valid_list: TXT file or list with the paths of the images to use for validating (Images must be between 0 and 255)
        test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        database_root: Path to the root of the Database
        Returns:
        """
        # Define types of data augmentation
        data_aug_rotates = [90, 180, 270]
        data_aug_flip = True

        # Load training images (path) and labels
        print('Started loading files...')
        if not isinstance(train_list, list) and train_list is not None:
            with open(train_list, 'r') as t:
                train_paths = t.readlines()
        elif isinstance(train_list, list):
            train_paths = train_list
        else:
            train_paths = []
        if not isinstance(valid_list, list) and valid_list is not None:
            with open(valid_list, 'r') as t:
                valid_paths = t.readlines();
        elif isinstance(valid_list, list):
            valid_paths = valid_list
        else:
            valid_paths = []
        if not isinstance(test_list, list) and test_list is not None:
            with open(test_list, 'r') as t:
                test_paths = t.readlines()
        elif isinstance(test_list, list):
            test_paths = test_list
        else:
            test_paths = []
        # Load training images and labels
        self.images_train = []
        self.labels_train = []
        for idx, line in enumerate(train_paths):
            img = Image.open(os.path.join(database_root, str(line.split('\t')[0])))
            img = img.resize(tuple([224, 224]))
            label = int(line.split('\t')[1])
            if data_aug:
                for angle in data_aug_rotates:
                    img_rot = img.rotate(angle)
                    self.images_train.append(np.array(img_rot, dtype=np.uint8))
                    self.labels_train.append(label)
                    if data_aug_flip:
                        img_rot_fl = img_rot.transpose(Image.FLIP_LEFT_RIGHT)
                        self.images_train.append(np.array(img_rot_fl, dtype=np.uint8))
                        self.labels_train.append(label)
            else:
                self.images_train.append(np.array(img, dtype=np.uint8))
                self.labels_train.append(label)
            if (idx + 1) % 1000 == 0:
                print('Loaded ' + str(idx) + ' train images') 
        # Load validating images and labels
        self.images_valid = []
        self.labels_valid = []
        for idx, line in enumerate(valid_paths):
            img = Image.open(os.path.join(database_root, str(line.split('\t')[0])))
            img = img.resize(tuple([224, 224]))
            label = int(line.split('\t')[1])
            if data_aug:
                for angle in data_aug_rotates:
                    img_rot = img.rotate(angle)
                    self.images_valid.append(np.array(img_rot, dtype=np.uint8))
                    self.labels_valid.append(label)
                    if data_aug_flip:
                        img_rot_fl = img_rot.transpose(Image.FLIP_LEFT_RIGHT)
                        self.images_valid.append(np.array(img_rot_fl, dtype=np.uint8))
                        self.labels_valid.append(label)
            else:
                self.images_valid.append(np.array(img, dtype=np.uint8))
                self.labels_valid.append(label)
            if (idx + 1) % 1000 == 0:
                print('Loaded ' + str(idx) + ' valid images')
        # Load testing images
        self.images_test = []
        for idx, line in enumerate(test_paths):
            img = Image.open(os.path.join(database_root, str(line.split('\t')[0])))
            img = img.resize(tuple([224, 224]))
            self.images_test.append(np.array(img, dtype=np.uint8))
            if (idx + 1) % 1000 == 0:
                print('Loaded ' + str(idx) + ' test images')
        print('Done initializing Dataset')

        # Init parameters
        self.train_ptr = 0
        self.valid_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.images_train)
        self.valid_size = len(self.images_valid)
        self.test_size = len(self.images_test)
        self.train_idx = np.arange(self.train_size)
        np.random.shuffle(self.train_idx)
        self.valid_idx = np.arange(self.valid_size)
        np.random.shuffle(self.valid_idx)

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
        batch_size: Size of the batch
        phase: Possible options:'train', 'valid' or 'test'
        Returns in training:
        images: List of Numpy arrays of the images
        labels: List of Numpy arrays of the labels
        Returns in validating:
        images: List of Numpy arrays of the images
        labels: List of Numpy arrays of the labels
        Returns in testing:
        images: Numpy array of the image
        """
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                images = [self.images_train[l] for l in idx]
                labels = [self.labels_train[l] for l in idx]
                self.train_ptr += batch_size
            else:
                old_idx = np.array(self.train_idx[self.train_ptr:])
                np.random.shuffle(self.train_idx)
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                idx = np.array(self.train_idx[:new_ptr])
                images_1 = [self.images_train[l] for l in old_idx]
                labels_1 = [self.labels_train[l] for l in old_idx]
                images_2 = [self.images_train[l] for l in idx]
                labels_2 = [self.labels_train[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                self.train_ptr = new_ptr
            return images, labels
        elif phase == 'valid':
            if self.valid_ptr + batch_size < self.valid_size:
                idx = np.array(self.valid_idx[self.valid_ptr:self.valid_ptr + batch_size])
                images = [self.images_valid[l] for l in idx]
                labels = [self.labels_valid[l] for l in idx]
                self.valid_ptr += batch_size
            else:
                old_idx = np.array(self.valid_idx[self.valid_ptr:])
                np.random.shuffle(self.valid_idx)
                new_ptr = (self.valid_ptr + batch_size) % self.valid_size
                idx = np.array(self.valid_idx[:new_ptr])
                images_1 = [self.images_valid[l] for l in old_idx]
                labels_1 = [self.labels_valid[l] for l in old_idx]
                images_2 = [self.images_valid[l] for l in idx]
                labels_2 = [self.labels_valid[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                self.train_ptr = new_ptr
            return images, labels
        elif phase == 'test':
            images = None
            if self.test_ptr + batch_size < self.test_size:
                images = self.images_test[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                images = self.images_test[self.test_ptr:] + self.images_test[:new_ptr]
                self.test_ptr = new_ptr
            return images
        else:
            return None

    def get_train_size(self):
        return self.train_size

    def get_valid_size(self):
        return self.valid_size

    def get_test_size(self):
        return self.test_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width
