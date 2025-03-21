#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.fft as fft

# class DisEntangle(nn.Module): # 입력 x 의 특정차에 대해 주파수 변환을 수행함. 
#     def __init__(self, in_channels):
#         super(DisEntangle, self).__init__()

#         self.in_channels = in_channels

#     def forward(self, x):
    
#         shape = x.shape
    
#         frequencies = fft.fftfreq(shape[2]).cuda() # 입력데이터의 3번째 차원 gpu로 보냄.
#         fft_compute = fft.fft(x, dim=2, norm='ortho').abs() # 입력 x에 대해 fft 계산, norm='ortho'> 정규직교모드 지정 , 절댓값 취함.
#          #  norm='ortho' > FFT와 역FFT(IFFT)에 각각 1/sqrt(N) 스케일링 인자 적용. 이런 방식을 통해 변환 수행 한 후에도 신호의 에너지 보존. 
#         frequencies = frequencies.unsqueeze(1) # 차원확장을 위해 연산 가능한 형태로 변환
#         frequencies = frequencies.unsqueeze(1)
#         frequencies = frequencies.unsqueeze(0)
#         frequencies = frequencies.unsqueeze(0) # 초기 차원이 (a,b)라고 하면, (1,1,a,1,1,b)
    
#         x = x * frequencies * frequencies * fft_compute * fft_compute # 특정 주파수 성분의 강도 조절 가능 (제곱을 취했기 때문)
    
#         return x

class DisEntangle(nn.Module):
    def __init__(self, in_channels):
        super(DisEntangle, self).__init__()

        self.in_channels = in_channels

    def forward(self, x):
        # 입력 x는 [batch_size, channels, height, width] 형태로 가정
        assert x.dim() == 4, f"Expected 4D input, but got {x.dim()}D input"

        # FFT는 높이(height)와 너비(width)에 대해 수행해야 함.
        batch_size, channels, height, width = x.shape

        # 2D FFT 수행 (높이와 너비에 대해)
        fft_compute = fft.fft2(x, dim=(-2, -1), norm='ortho').abs()

        # 높이와 너비에 대한 주파수 계산
        freq_h = fft.fftfreq(height).to(x.device)  # 높이에 대한 주파수 계산
        freq_w = fft.fftfreq(width).to(x.device)   # 너비에 대한 주파수 계산

        # 주파수 차원 확장 (배치와 채널 차원에 맞게 브로드캐스팅)
        freq_h = freq_h.unsqueeze(1).unsqueeze(0).unsqueeze(0)  # [1, 1, height, 1]
        freq_w = freq_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, width]

        # 주파수 성분을 이미지에 곱함 (높이, 너비 주파수 성분의 제곱)
        frequency_weights = (freq_h ** 2 + freq_w ** 2).sqrt()  # 주파수의 크기를 구함
        # frequency_weights = frequency_weights.to(DEVICE)  # GPU로 전송

        # 주파수 성분을 조정하여 곱함
        x = x * frequency_weights * fft_compute

        return x
        
# class SpatialCausality(nn.Module): # fft 사용.
#     def __init__(self, in_channels):
#         super(SpatialCausality, self).__init__()
#         self.in_channels = in_channels
    
#     def forward(self, x):
    
#         shape = x.shape 
#         fft_compute = fft.fft2(x.view(shape[0], shape[1], shape[2], shape[3]*shape[4]),dim=(2,3), norm='ortho')
#          # 2차원 fft 구함.
#         fft_compute = torch.conj(fft_compute) * fft_compute # Correlation
#          # 신호의 상관관계를 구함
#         fft_compute = fft.ifft2(fft_compute, dim=(2,3), norm='ortho') # Inverse FFT. 상관관계 결과에 역FFT 적용(IFFT)
#         fft_compute = fft_compute.view(shape[0], shape[1], shape[2], shape[3], shape[4]) # Reshape  
    
#         x = x + 0.01 * fft_compute.abs() * x  #특징 강조를 위해 작은 계수를 곱함.?

#         return x #0.01 * torch.real(fft_compute)

class SpatialCausality(nn.Module):
    def __init__(self, in_channels):
        super(SpatialCausality, self).__init__()
        self.in_channels = in_channels
    
    def forward(self, x):
        # 입력 텐서의 형태를 확인
        shape = x.shape
        
        # 입력 데이터가 4D 텐서인지 확인
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        # 이미지의 공간적 차원에 대해 FFT를 수행
        # x.view(shape[0], shape[1], shape[2], shape[3])로 4D 텐서 유지
        fft_compute = fft.fft2(x, dim=(2, 3), norm='ortho')
        fft_compute = torch.conj(fft_compute) * fft_compute  # 상관관계 계산
        fft_compute = fft.ifft2(fft_compute, dim=(2, 3), norm='ortho')  # 역 FFT
        fft_compute = fft_compute.real  # 복소수에서 실수 부분만 추출

        # 특징 강조를 위해 입력 이미지에 강조 계수를 곱함
        x = x + 0.01 * fft_compute.abs() * x
        
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.objback_disentangle = DisEntangle(in_channels=256)
        self.spatialCaus = SpatialCausality(in_channels=256)

        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        low_level_feat = x
        x = self.objback_disentangle(x)
        
        x = self.layer2(x)
        
        x = self.spatialCaus(x) + self.objback_disentangle(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FFT_DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructed FFT_DeepLabv3_plus model.")
            # print("Number of classes: {}".format(n_classes))
            # print("Output stride: {}".format(os))
            # print("Number of Input Channels: {}".format(nInputChannels))
        super(FFT_DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


# if __name__ == "__main__":
#     model = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True)
#     model.eval()
#     image = torch.randn(1, 3, 512, 512)
#     with torch.no_grad():
#         output = model.forward(image)
#     print(output.size())