import torch
import torch.nn as  nn
from torch.nn.modules import dropout

initial_in_channels = 3

class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, n_class=1000):
        super(GoogleNet, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception_Block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_Block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_Block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_Block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_Block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_Block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_Block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_Block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_Block(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, n_class)   

        self.aux_1 = Aux_Block(512, n_class)
        self.aux_2 = Aux_Block(528, n_class)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool1(x)
        x = self.conv_2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        aux1 = self.aux_1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2 = self.aux_2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        print(f'FC1 Block: {x.shape}')
        x = self.fc1(x)

        return x, aux1, aux2

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_1x1_pool):
        super(Inception_Block, self).__init__()

        self.block_1 = nn.Sequential(
            Conv_Block(in_channels, out_1x1, kernel_size=1)
        )

        self.block_2 = nn.Sequential(
            Conv_Block(in_channels, reduce_3x3, kernel_size=1),
            Conv_Block(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.block_3 = nn.Sequential(
            Conv_Block(in_channels, reduce_5x5, kernel_size=1),
            Conv_Block(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv_Block(in_channels, out_1x1_pool, kernel_size=1)
        )
    
    def forward(self, x):
        return torch.cat([self.block_1(x), self.block_2(x), self.block_3(x), self.block_4(x)], axis=1)

class Aux_Block(nn.Module):
    def __init__(self, in_channels, n_class):
        super(Aux_Block, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=3)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1152, 1024)
        self.fc2 = nn.Linear(1024, n_class)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        print(f'Aux Block: {x.shape}')
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride=1, padding=0):
        super(Conv_Block, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
    
    def forward(self, image):
        image = self.cnn(image)
        image = self.relu(image)
        return image

if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    model = GoogleNet()
    print(model(x)[0].shape)