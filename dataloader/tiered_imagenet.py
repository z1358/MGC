import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
IMAGE_PATH1 = osp.join(ROOT_PATH, 'data/tieredimagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/tieredimagenet/split')


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class tieredImageNet(Dataset):
    def __init__(self, setname, args, augment=False):
        assert (setname == 'train' or setname == 'val' or setname == 'test')
        self.netdict = {}
        self.setname = setname
        if setname == 'train':
            with open(osp.join(SPLIT_PATH, 'coarse.txt')) as words:
                for line in words:
                    self.netdict[line.strip('\n').split('\t')[0]] = line.strip('\n').split('\t')[1]

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        print("THIS_PATH : ", THIS_PATH )
        print("ROOT_PATH: ", ROOT_PATH )

        self.data, self.label, self.coarse = self.parse_csv(csv_path, setname)

        self.num_class = len(set(self.label))
        if setname == 'train':
            self.coarse_class = len(set(self.coarse))

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'ResNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        coarse_label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH1, name)
            if setname == 'train':
                coarse = int(self.netdict[wnid])
                coarse_label.append(coarse)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        if setname == 'train':
            return data, label, coarse_label
        return data, label, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))
        if self.setname == 'train':
            coarse_label = self.coarse[i]
            return image, label, coarse_label
        return image, label

