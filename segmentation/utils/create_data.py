# dataset
from utils.augmentation import augmentation_train, augmentation_test
from utils.dataset import myDataSet,Test_Detection

import glob
import os

def create_dataset(data_name = ''):
    train_images = sorted(glob.glob("/content/drive/MyDrive/sejong/segmentation/data/images/train/*"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    train_labels = sorted(glob.glob("/content/drive/MyDrive/sejong/segmentation/data/labels/train/*"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    train_images = [img for img in train_images if img.find('jpg')!= -1] # super pixels 이미지 제외

    valid_images = sorted(glob.glob("/content/drive/MyDrive/sejong/segmentation/data/images/val/*"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    valid_labels = sorted(glob.glob("/content/drive/MyDrive/sejong/segmentation/data/labels/val/*"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    valid_images = [img for img in valid_images if img.find('jpg')!= -1] # super pixels 이미지 제외
    
    test_images = sorted(glob.glob("/content/drive/MyDrive/sejong/segmentation/data/images/test/*"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    test_labels = sorted(glob.glob("/content/drive/MyDrive/sejong/segmentation/data/labels/test/*"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    test_images = [img for img in test_images if img.find('jpg')!= -1] # super pixels 이미지 제외
    

    train_images = sorted(train_images)
    train_labels = sorted(train_labels)

    valid_images = sorted(valid_images)
    valid_labels = sorted(valid_labels)

    test_images = sorted(test_images)
    test_labels = sorted(test_labels)

    
    # 데이터셋 클래스 적용
    train_transforms = augmentation_train()
    test_transforms = augmentation_test()

    custom_dataset_train = myDataSet(train_images, train_labels, transforms=train_transforms)
    # print("My custom training-dataset has {} elements".format(len(custom_dataset_train)))

    custom_dataset_val = myDataSet(valid_images, valid_labels, transforms=test_transforms)
    # print("My custom valing-dataset has {} elements".format(len(custom_dataset_val)))
    
    custom_dataset_test = myDataSet(test_images, test_labels, transforms=test_transforms)
    # print("My custom valing-dataset has {} elements".format(len(custom_dataset_val)))
    
    return custom_dataset_train, custom_dataset_val, custom_dataset_test


def create_detection_dataset(data_name = ''):
    # test_detection_images = sorted(glob.glob("/content/drive/MyDrive/sejong/ex/detection/images/*.jpg"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    test_detection_images = sorted(glob.glob("/content/drive/MyDrive/sejong/far/data/data_1/detection/images/*.jpg"),key = lambda x: int(x.split('/')[-1].split('.')[0]))
    # test_detection_images = sorted(glob.glob("/content/drive/MyDrive/sejong/ex_detection/images/*.jpg"), key=lambda x: os.path.splitext(x)[0])
    # test_detection_images = [img for img in test_detection_images if img.find('jpg')!= -1] # super pixels 이미지 제외
    # test_detection_images = sorted(test_detection_images)
    
    test_transforms = augmentation_test()
    
    test_detection_img = Test_Detection(test_detection_images, transforms=test_transforms)
    
    return test_detection_img