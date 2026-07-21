import torch

def create_model(model_name):    
    if model_name == "fcbformer":
        from models.FCBmodels import FCBFormer
        model = FCBFormer(size=224)
        return model
    
    elif model_name == "unet":
        from models.UNet import UNet
        model = UNet(n_classes = 1)
        return model
        
    elif model_name == "deeplab_v3_p":
        from models.DeepLabv3_plus import DeepLabv3_plus
        model = DeepLabv3_plus(n_classes=1,os=8, pretrained=True)
        return model

    elif model_name == "fft_deeplab_v3_p":
        from models.FFT_DeepLabv3_plus import FFT_DeepLabv3_plus
        model = FFT_DeepLabv3_plus(n_classes=1,os=8, pretrained=True)
        return model
