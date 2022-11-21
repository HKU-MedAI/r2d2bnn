from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from PIL import Image
import os


class MVTecDataset(Dataset):
    def __init__(self, img_dir: str, image_size: int, train=False, valid=False, test=False) -> None:
        self.labels = []
        self.data_paths = []
        self.gt_paths = []
        self.train = train
        self.valid = valid
        self.test = test

        if self.train:
            # train the pretrain model
            for i, cl in enumerate(os.listdir(img_dir)):
                path = os.path.join(img_dir, cl, 'train', 'good')
                img_list = Path(path)
                self.data_paths.extend([p for p in img_list.iterdir()])
                self.labels.extend([i for _ in img_list.iterdir()])
        elif self.valid:
            for i, cl in enumerate(os.listdir(img_dir)):
                path = os.path.join(img_dir, cl, 'test', 'good')
                img_list = Path(path)
                self.data_paths.extend([p for p in img_list.iterdir()])
                self.labels.extend([i for _ in img_list.iterdir()])
        # if train:
        #     self.data_paths.extend([p for p in Path(img_dir).iterdir()])
        else:  # eval
            img_dir = Path(img_dir)
            self.data_paths = [p for p in img_dir.iterdir()]

            # get the GT mask
            # for img_path in self.data_paths:
            #     if 'good' not in img_path:
            #         path, file_name = os.path.split(img_path)
            #         mask_file_name = file_name.split('.png')[0] + '_mask.png'
            #         mask_path = os.path.join('/'.join(path.split('/')[:-2]), 'ground_truth', path.split('/')[-1], mask_file_name)
            #         self.gt_paths.append(mask_path)
            #     else:
            #         self.gt_paths.append('None')

            for path in img_dir.iterdir():
                if 'good' in str(path):
                    self.labels.append(0)
                else:
                    self.labels.append(1)

        if self.train:
            transform_list = [Resize(image_size),
                              transforms.RandomApply([
                                  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                                  transforms.ColorJitter(brightness=0.5,
                                                         contrast=0.5,
                                                         saturation=0.5,
                                                         hue=0),
                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.RandomHorizontalFlip(),
                              ], p=0.8),
                              Cutout(n_holes=2, length=100),
                              CutPasteScar(),
                              transforms.ToTensor()]
            self.anc_transform = transforms.Compose([transform_list[0]] + [transforms.ToTensor()])
            self.pos_transform = transforms.Compose(transform_list[:1] + [transforms.ToTensor()])
            self.neg_transform = transforms.Compose(transform_list[:3] + [transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([Resize(image_size),
                                                 transforms.ToTensor()])

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.data_paths[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        label = self.labels[idx]

        # if "good" in str(img_path):
        #     label = 1
        # else:
        #     label = 0

        # if self.train:
        #     if idx < len(self.data_paths)/2:
        #         label = 1
        #     else:
        #         label = 0
        # else:
        #     if "good" in str(img_path):
        #         label = 1
        #     else:
        #         label = 0

        # if "broken" in str(img_path):
        #     images_tf = img.numpy().transpose((1,2,0)) * 255
        #     images_tf = cv2.cvtColor(images_tf,cv2.COLOR_RGB2BGR)
        #     img_name = 'train_image/' + str(idx) + '.png'
        #     cv2.imwrite(img_name, images_tf)

        return img, label

    def __len__(self) -> int:
        return len(self.data_paths)


class MVTecTrainDataset(Dataset):
    def __init__(self, img_dir: str, cl: str, image_size: int) -> None:
        self.labels = []
        self.data_paths = []
        self.gt_paths = []

        # train the encoder

        path = os.path.join(img_dir, cl, 'train', 'good')
        img_list = Path(path)
        self.data_paths.extend([p for p in img_list.iterdir()])

        transform_list = [
            Resize(image_size),
            transforms.RandomApply([
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                transforms.ColorJitter(brightness=0.5,
                                       contrast=0.5,
                                       saturation=0.5,
                                       hue=0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
            ], p=0.8),
            Cutout(n_holes=1, length=50),
            CutPasteUnion(),
            transforms.Lambda(lambda crops: transforms.ToTensor()(crops[1]))
        ]
        self.anc_transform = transforms.Compose([transform_list[0]] + [transforms.ToTensor()])
        self.pos_transform = transforms.Compose(transform_list[:2] + [transforms.ToTensor()])
        self.neg_transform = transforms.Compose(transform_list)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.data_paths[idx]
        img = Image.open(img_path).convert('RGB')

        img_anc = self.anc_transform(img).type(torch.FloatTensor)
        img_pos = self.pos_transform(img).type(torch.FloatTensor)
        img_neg = self.neg_transform(img).type(torch.FloatTensor)

        return img_anc, img_pos, img_neg

    def __len__(self) -> int:
        return len(self.data_paths)


class MVTecTestDataset(Dataset):
    def __init__(self, img_dir: str, gt_dir: str, image_size: int) -> None:
        self.labels = []

        img_dir = Path(img_dir)
        gt_dir = Path(gt_dir)
        self.data_paths = [p for p in img_dir.iterdir()]

        if "good" in str(gt_dir):
            self.gt_paths = None
        else:
            self.gt_paths = [p for p in gt_dir.iterdir()]

        for path in img_dir.iterdir():
            if 'good' in str(path):
                self.labels.append(0)
            else:
                self.labels.append(1)

        self.transform = transforms.Compose([
            Resize(image_size),
            transforms.ToTensor()])

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.data_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = img.type(torch.FloatTensor)

        if self.gt_paths:
            gt_path = self.gt_paths[idx]
            img_gt = Image.open(gt_path).convert("L")
            img_gt = self.transform(img_gt)
            img_gt = img_gt.type(torch.FloatTensor)
        else:
            img_gt = torch.zeros(img.shape)

        label = self.labels[idx]

        return img, img_gt, label

    def __len__(self) -> int:
        return len(self.data_paths)


class PatchDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)
