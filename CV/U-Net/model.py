import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down_conv1 = DoubleConv(1, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.down_conv5 = DoubleConv(512, 1024)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)

        self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)

        self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)

        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        self.final = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
    
    def forward(self, x):
        #Encoder
        x1 = self.down_conv1(x) #For Skip Connection
        x2 = self.max_pool(x1)
        x3 = self.down_conv2(x2) #For Skip Connection
        x4 = self.max_pool(x3)
        x5 = self.down_conv3(x4) #For Skip Connection
        x6 = self.max_pool(x5)
        x7 = self.down_conv4(x6) #For Skip Connection
        x8 = self.max_pool(x7)
        x9 = self.down_conv5(x8)

        #decoder
        x = self.up_trans1(x9)
        x7_re = F.resize(x7, (x.shape[2],x.shape[3]))
        x = torch.cat([x, x7_re], axis=1)
        x = self.up_conv1(x)

        x = self.up_trans2(x)
        x5_re = F.resize(x5, (x.shape[2],x.shape[3]))
        x = torch.cat([x, x5_re], axis=1)
        x = self.up_conv2(x)

        x = self.up_trans3(x)
        x3_re = F.resize(x3, (x.shape[2],x.shape[3]))
        x = torch.cat([x, x3_re], axis=1)
        x = self.up_conv3(x)

        x = self.up_trans4(x)
        x1_re = F.resize(x1, (x.shape[2],x.shape[3]))
        x = torch.cat([x, x1_re], axis=1)
        x = self.up_conv4(x)

        x = self.final(x)

        return x


if __name__ == "__main__":
    image = torch.randn(1, 1, 572, 572)
    model = Unet()
    print(model(image).shape)
