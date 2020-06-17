import torch
import torch.nn as nn
from vgg import vgg19

class VGG19(torch.nn.Module):
    """
    VGG19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    level1: 64*2=128; level2: 128*2=256; level3: 256*4=1024; level4: 512*4=2048; level5: 512*4=2048
    Total dimension: 128 + 256 + 1024 + 2048 + 2048 = 5504
    """
    def __init__(self, gradient=False):
        super(VGG19, self).__init__()
        features = vgg19(pretrained=True).features    # feature layers
        """ vgg.features
        Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace)                                                        # self.relu1_1
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace)                                                        # self.relu1_2
          
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (6): ReLU(inplace)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace)  
              
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (11): ReLU(inplace)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace)
          (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (17): ReLU(inplace)
          
          (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU(inplace)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace)
          (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (24): ReLU(inplace)
          (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): ReLU(inplace)
          
          (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (29): ReLU(inplace)
          (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (31): ReLU(inplace)
          (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (33): ReLU(inplace)
          (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (35): ReLU(inplace)
          
          (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        """
        # hierarchy 1 (level 1)
        self.conv1_1 = features[0]
        self.relu1_1 = features[1]
        self.conv1_2 = features[2]
        self.relu1_2 = features[3]

        # hierarchy 2 (level 2)
        self.pool1 = features[4]
        self.conv2_1 = features[5]
        self.relu2_1 = features[6]
        self.conv2_2 = features[7]
        self.relu2_2 = features[8]

        # hierarchy 3 (level 3)
        self.pool2 = features[9]
        self.conv3_1 = features[10]
        self.relu3_1 = features[11]
        self.conv3_2 = features[12]
        self.relu3_2 = features[13]
        self.conv3_3 = features[14]
        self.relu3_3 = features[15]
        self.conv3_4 = features[16]
        self.relu3_4 = features[17]

        # hierarchy 4 (level 4)
        self.pool3 = features[18]
        self.conv4_1 = features[19]
        self.relu4_1 = features[20]
        self.conv4_2 = features[21]
        self.relu4_2 = features[22]
        self.conv4_3 = features[23]
        self.relu4_3 = features[24]
        self.conv4_4 = features[25]
        self.relu4_4 = features[26]

        # hierarchy 5 (level 5)
        self.pool4 = features[27]
        self.conv5_1 = features[28]
        self.relu5_1 = features[29]
        self.conv5_2 = features[30]
        self.relu5_2 = features[31]
        self.conv5_3 = features[32]
        self.relu5_3 = features[33]
        self.conv5_4 = features[34]
        self.relu5_4 = features[35]

        self.pool5 = features[36]

        # don't need the gradients, just want the features
        if not gradient:
            for param in self.parameters():
                param.requires_grad = False

        self.pad = nn.ReflectionPad2d(padding=1)

    def forward(self, x, feature_layers):
        # level 1
        x = self.pad(x)
        conv1_1 = self.conv1_1(x)
        relu1_1 = self.pad(self.relu1_1(conv1_1))
        conv1_2 = self.conv1_2(relu1_1)
        relu1_2 = self.relu1_2(conv1_2)
        pool1 = self.pool1(relu1_2)

        # level 2
        pool1 = self.pad(pool1)
        conv2_1 = self.conv2_1(pool1)
        relu2_1 = self.pad(self.relu2_1(conv2_1))
        conv2_2 = self.conv2_2(relu2_1)
        relu2_2 = self.relu2_2(conv2_2)
        pool2 = self.pool2(relu2_2)

        # level 3
        pool2 = self.pad(pool2)
        conv3_1 = self.conv3_1(pool2)
        relu3_1 = self.pad(self.relu3_1(conv3_1))
        conv3_2 = self.conv3_2(relu3_1)
        relu3_2 = self.pad(self.relu3_2(conv3_2))
        conv3_3 = self.conv3_3(relu3_2)
        relu3_3 = self.pad(self.relu3_3(conv3_3))
        conv3_4 = self.conv3_4(relu3_3)
        relu3_4 = self.relu3_4(conv3_4)
        pool3 = self.pool3(relu3_4)

        # level 4
        pool3 = self.pad(pool3)
        conv4_1 = self.conv4_1(pool3)
        relu4_1 = self.pad(self.relu4_1(conv4_1))
        conv4_2 = self.conv4_2(relu4_1)
        relu4_2 = self.pad(self.relu4_2(conv4_2))
        conv4_3 = self.conv4_3(relu4_2)
        relu4_3 = self.pad(self.relu4_3(conv4_3))
        conv4_4 = self.conv4_4(relu4_3)
        relu4_4 = self.relu4_4(conv4_4)
        pool4 = self.pool4(relu4_4)

        # level 5
        pool4 = self.pad(pool4)
        conv5_1 = self.conv5_1(pool4)
        relu5_1 = self.pad(self.relu5_1(conv5_1))
        conv5_2 = self.conv5_2(relu5_1)
        relu5_2 = self.pad(self.relu5_2(conv5_2))
        conv5_3 = self.conv5_3(relu5_2)
        relu5_3 = self.pad(self.relu5_3(conv5_3))
        conv5_4 = self.conv5_4(relu5_3)
        relu5_4 = self.relu5_4(conv5_4)
        # pool5 = self.pool5(relu5_4)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return dict((key, value) for key, value in out.items() if key in feature_layers)
