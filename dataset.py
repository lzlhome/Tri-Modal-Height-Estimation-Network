import copy
import os
import osgeo.gdal as gdal
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms as T

def floor2height(sample):
    # input:  gdal.Open(self.data[0]).ReadAsArray()    [c, h, w]
    new_matrix = sample
    new_matrix[0, :, :] = new_matrix[0, :, :] * 3

    return new_matrix


class diydataset(data.Dataset):
    def __init__(self, root="./data/ImageSets", transforms=None, augmentation=None, txt_name: str = "train.txt", is_pred=False):
        super(diydataset, self).__init__()
        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(txt_path, "r") as f:
            self.data = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.is_pred = is_pred

        if transforms is None:
            self.transforms = self.dataPreset()
        else:
            self.transforms = transforms
            if self.is_pred:
                new_trans = copy.deepcopy(self.transforms)
                self.transforms.transforms[1].mean = new_trans.transforms[1].mean[1:]
                self.transforms.transforms[1].std = new_trans.transforms[1].std[1:]

        self.augmentation = augmentation


    def dataPreset(self):
        '''
        RandomCrop to 224   先裁剪再归一化
        gdal.Open(self.data[0]).ReadAsArray(): [c, h, w]
        ToTensor()->[w, c, h]
        .permute(1, 2, 0)->[c, h, w]
        '''

        sample_list = []
        for item in self.data:
            img = T.ToTensor()(gdal.Open(item).ReadAsArray()).permute(1, 2, 0)
            sample_list.append(img.unsqueeze(0))
        concatenated_tensor = torch.cat(sample_list, dim=0)

        mean = torch.mean(concatenated_tensor, dim=(0, 2, 3))
        std = torch.std(concatenated_tensor, dim=(0, 2, 3))
        std[std == 0] = 1e-5

        trans = [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

        return T.Compose(trans)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        tensor_sample = self.transforms(np.transpose(gdal.Open(self.data[index]).ReadAsArray(), (1, 2, 0)))

        temp_img = tensor_sample[:-1]  # 删去不需要的WSF

        if self.augmentation is not None:
            temp_img = self.augmentation(temp_img)

        if self.is_pred:
            img = temp_img
            target = []

            # get GetGeoTransform() and GetProjection() of sample
            dataset = gdal.Open(self.data[index])
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()
            return img, target, [geotransform, projection]

        else:
            img = temp_img[1:]
            target = temp_img[0].unsqueeze(0)
            return img, target

    def __len__(self):
        return len(self.data)

    def get_samples(self):
        """
        Returns:
            tensor: [num_samples, num_channels, h, w]
        """
        normalized_samples = []
        for item in self.data:
            tensor_sample = self.transforms(np.transpose(gdal.Open(item).ReadAsArray(), (1, 2, 0)))
            temp_img = tensor_sample[:-1]  # delete WSF
            normalized_samples.append(temp_img)
        samples_tensor = torch.cat([x.unsqueeze(0) for x in normalized_samples], dim=0)

        return samples_tensor
