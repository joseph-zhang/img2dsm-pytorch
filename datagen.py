import random
import torch
import numpy as np
from Dataloader import DFCloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


def setNanToZero(input, target):
    nanMask = np.isnan(target)

    _input = input.copy()
    _target = target.copy()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target


class depthDataset(Dataset):
    """ Dataset class for satellite image """

    def __init__(self, opt, mode, transform=None, target_transform=None):
        self.dfcloader = DFCloader(opt.dataset_dir, mode)
        self.dfcloader.get_pair_data()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        #target_transform = transforms.ToTensor()
        src_transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.ToTensor()])
        image, depth = self.dfcloader.load_item(idx)
        image, depth = setNanToZero(image, depth)
        src_image = src_transform(image.copy().astype(np.uint8))

        if self.transform:
            image = self.transform(image.astype(np.uint8))
        if self.target_transform:
            depth = self.target_transform(depth.astype(np.float32))
        else:
            pass

        sample = {'image':image, 'depth':depth, 'src':src_image}
        return sample

    def __len__(self):
        return self.dfcloader.get_data_length()


def getTrainingData(opt, batch_size):
    transformed_training = depthDataset(opt,
                                        "train_data",
                                        transform=transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.ToTensor(),
                                        ]),
                                        target_transform=transforms.ToTensor())

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=0, pin_memory=False)

    return dataloader_training


class depthDataset_test(Dataset):
    """ Dataset class for satellite image """

    def __init__(self, opt, mode, transform=None):
        self.dfcloader = DFCloader(opt.dataset_dir, mode, 1024, 1024, 512, 512, 256)
        self.dfcloader.get_pair_data()
        self.transform = transform

    def __getitem__(self, idx):
        target_transform = transforms.ToTensor()
        if self.transform:
            image, depth = self.dfcloader.load_item(idx)
            image, depth = setNanToZero(image, depth)
            src_image = image.copy().astype(np.uint8)
            image = self.transform(image.astype(np.uint8))
            depth = target_transform(depth).float()

        sample = {'image':image, 'depth':depth, 'src':src_image}
        return sample

    def __len__(self):
        return len(self.dfcloader.top_data)


def getTestingData(opt, batch_size=1):
    transformed_training = depthDataset_test(opt,
                                             "test_data",
                                             transform=transforms.Compose([
                                                 transforms.ToPILImage(),
                                                 transforms.Resize(256),
                                                 transforms.ToTensor(),
                                             ]))

    dataloader_testing = DataLoader(transformed_training, batch_size,
                                     shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
