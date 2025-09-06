import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as transforms

from .tools import default_loader, mask_loader, ground_truth_loader, is_image_file, normalize


class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, data_aug):
        super(Dataset, self).__init__()
        self.ground_truth = [x for x in os.listdir(f"{data_path}/full") if is_image_file(x)]
        self.partial_map = [x for x in os.listdir(f"{data_path}/part") if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.data_aug = data_aug
        self.image_raw_shape = default_loader(os.path.join(f"{data_path}/full", self.ground_truth[0])).size

    @staticmethod
    def rotate_img(img, rot):
        rotations = {
            0: lambda x: x,
            90: lambda x: x.transpose(Image.ROTATE_90),
            180: lambda x: x.transpose(Image.ROTATE_180),
            270: lambda x: x.transpose(Image.ROTATE_270)
        }
        return rotations[rot](img)

    def crop_img(self, path, img_type=None, rotation=0):
        if img_type == 'belief':
            img = default_loader(path)
            img = img.convert('L')
        elif img_type == 'mask':
            img = mask_loader(path)
            img = img.convert('1')
        elif img_type == 'gt':
            img = ground_truth_loader(path)
            img = img.convert('L')
        else:
            raise ValueError

        img_raw = img.copy()
        img = self.rotate_img(img, rotation)  # augmentation

        if self.image_raw_shape[0] < self.image_shape[0] and self.image_raw_shape[1] < self.image_shape[1]:
            pad_left = (self.image_shape[0] - self.image_raw_shape[0]) // 2
            pad_top = (self.image_shape[1] - self.image_raw_shape[1]) // 2
            pad_right = self.image_shape[0] - self.image_raw_shape[0] - pad_left
            pad_bottom = self.image_shape[1] - self.image_raw_shape[1] - pad_top
            img = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), padding_mode='edge')(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
        # img = transforms.RandomCrop(self.image_shape)(img)
        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img_raw = transforms.ToTensor()(img_raw)
        if img_type != 'mask':
            img = normalize(img)
            img_raw = normalize(img_raw)
        return img, img_raw

    def __getitem__(self, index):
        map_name = '_'.join(self.partial_map[index].split('_')[:-1])
        data_aug = random.choice([0, 90, 180, 270]) if self.data_aug else 0

        partial_img, partial_img_raw = self.crop_img(os.path.join(f"{self.data_path}/part", self.partial_map[index]), img_type='belief', rotation=data_aug)
        mask_img, mask_img_raw = self.crop_img(os.path.join(f"{self.data_path}/part", self.partial_map[index]), img_type='mask', rotation=data_aug)
        ground_truth, ground_truth_raw = self.crop_img(os.path.join(f"{self.data_path}/full", f"{map_name}.png"), img_type='gt', rotation=data_aug)

        if 'room' in map_name:
            index = torch.tensor(0)
        elif 'tunnel' in map_name:
            index = torch.tensor(1)
        elif 'outdoor' in map_name:
            index = torch.tensor(2)
        else:
            raise ValueError(f"Unknown map name: {map_name}")
        map_onehot = F.one_hot(index, num_classes=3).float()

        return ground_truth, partial_img, mask_img, map_onehot, (ground_truth_raw, partial_img_raw, mask_img_raw)

    def __len__(self):
        return len(self.partial_map)
