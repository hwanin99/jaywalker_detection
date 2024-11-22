from torchvision import transforms

def augmentation_train():
    _size = 224, 224
    resize = transforms.Resize(_size, interpolation=0)

    # set your transforms
    train_transforms = transforms.Compose([
                               transforms.Resize(_size, interpolation=0),
                               transforms.RandomRotation(180),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(_size, padding = 10), # needed after rotation (with original size)
                           ])
    return train_transforms




def augmentation_test():
    _size = 224, 224
    test_transforms = transforms.Compose([
                               transforms.Resize(_size, interpolation=0),
                           ])
    return test_transforms
    
    
def detection_aug_train():
    _size = 448, 448
    train_transforms = transforms.Compose([
                                transforms.Resize(_size, interpolation=0),
                                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 2.0))
                                ])
    return train_transforms

