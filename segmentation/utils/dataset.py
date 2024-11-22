import os
import imageio

import numpy as np
import torch
import torchvision.transforms as transforms
import glob
import cv2

class myDataSet(object):

    def __init__(self, path_images, path_masks, transforms):
        "Initialization"
        self.all_path_images = sorted(path_images)
        self.all_path_masks = sorted(path_masks)
        self.transforms = transforms

    def __len__(self):
        "Returns length of dataset"
        return len(self.all_path_images)

    def __getitem__(self, index):
        "Return next item of dataset"

        if torch.is_tensor(index):        # 인덱스가 tensor 형태일 수 있으니 리스트 형태로 바꿔준다.
            index = index.tolist()

        # Define path to current image and corresponding mask
        path_img = self.all_path_images[index]
        path_mask = self.all_path_masks[index]

        # Load image and mask:
        #     .jpeg has 3 channels, channels recorded last
        #     .jpeg records values as intensities from 0 to 255
        #     masks for some reason have values different to 0 or 255: 0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255
        img_bgr = cv2.imread(path_img)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cv2는 채널이 BGR로 저장된다 -> 출력할 때 RGB로 바꿔줘야함
        img = img / 255  # 픽셀 값들을 0~1로 변환한다

        # mask = cv2.imread(path_mask)[:, :, 0] / 255  # 마스크의 채널은 1개만 있으면 된다
        # mask = mask.round()
        mask = cv2.imread(path_mask)[:, :, 0]
        mask[mask == 2] = 1  # 횡단보도(class:2)를 class:1로 변경
        mask[mask != 1] = 0  # 1이 아닌 값은 0으로 변경

        # note, resizing happens inside transforms

        # convert to Tensors and fix the dimentions
        img = torch.FloatTensor(np.transpose(img, [2, 0 ,1])) # Pytorch uses the channels in the first dimension
        mask = torch.FloatTensor(mask).unsqueeze(0) # Adding channel dimension to label
        # mask = torch.FloatTensor(np.transpose(mask, [2, 0 ,1]))



        # apply transforms/augmentation to both image and mask together
        sample = torch.cat((img, mask), 0) # insures that the same transform is applied
        sample = self.transforms(sample)

        img = sample[:img.shape[0], ...]
        mask = sample[img.shape[0]:, ...]

        return img, mask
        
        
class Test_Detection(object):

    def __init__(self, path_images,transforms):
        "Initialization"
        self.all_path_images = sorted(path_images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.transforms = transforms

    def __len__(self):
        "Returns length of dataset"
        return len(self.all_path_images)

    def __getitem__(self, index):
        "Return next item of dataset"

        if torch.is_tensor(index):        # 인덱스가 tensor 형태일 수 있으니 리스트 형태로 바꿔준다.
            index = index.tolist()

        # Define path to current image
        path_img = self.all_path_images[index]

        img_bgr = cv2.imread(path_img)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cv2는 채널이 BGR로 저장된다 -> 출력할 때 RGB로 바꿔줘야함
        img = img / 255  # 픽셀 값들을 0~1로 변환한다

        # convert to Tensors and fix the dimentions
        img = torch.FloatTensor(np.transpose(img, [2, 0 ,1])) # Pytorch uses the channels in the first dimension
        img =self.transforms(img)

        return img

