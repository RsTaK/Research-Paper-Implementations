import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

VGG_TYPE = {
    'vgg11' : [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'vgg13' : [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'vgg16' : [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
    'vgg19' : [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'pool']
}

CONFIG = {
    'start_in_channel' : 3,
    'cnn_kernel' : (3,3),
    'cnn_stride' : (1,1),
    'cnn_padding' : (1,1),

    'pool_kernel' : (2, 2),
    'pool_stride' : (2,2)
}

class VGG(nn.Module):
    def __init__(self, img_size = (224, 224), n_classes=1000, arch='vgg16'):
        super(VGG, self).__init__()
        self.cnn, pool_count = self.create_cnn_layers(VGG_TYPE[arch])
        h_out = img_size[0]//pow(2,pool_count)
        w_out = img_size[1]//pow(2,pool_count)
        self.fc = self.create_fn_layers(h_out, w_out, n_classes)

    def forward(self, image):
        image = self.cnn(image)
        image = image.view(image.size(0), -1)
        image = self.fc(image)
        return image

    def create_cnn_layers(self, vgg_config):
        layers = list()
        pool_count = 0
        in_channels = CONFIG['start_in_channel']

        for each in vgg_config:

            if type(each) == int:

                out_channels = each
                layers.extend([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=CONFIG['cnn_kernel'], stride=CONFIG['cnn_stride'], padding=CONFIG['cnn_padding']),
                nn.ReLU()])
                in_channels = out_channels

            else:
                layers.extend([nn.AvgPool2d(kernel_size=CONFIG['pool_kernel'], stride=CONFIG['pool_stride'])])
                pool_count += 1
        return nn.Sequential(*layers), pool_count

    def create_fn_layers(self, h_out, w_out, n_classes):
        fc = nn.Sequential(
            nn.Linear(512 * h_out* w_out, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_classes)    
        )
        return fc

if __name__=="__main__":
    sample = torch.randn(1, 3, 224, 224)
    model = VGG(img_size = (sample.size(2), sample.size(3)), arch='vgg19')

    print(model)
    print(model(sample).shape)