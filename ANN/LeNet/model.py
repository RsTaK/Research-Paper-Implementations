import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet, self).__init__()

        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0)),
        )

        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0)),
        )

        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            nn.Tanh(),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes)
        )
    
    def forward(self, image):

        image = self.cnn_block_1(image)
        image = self.cnn_block_2(image)
        image = self.cnn_block_3(image)

        image = image.view(image.size(0), -1)

        image = self.fc_block(image)

        return image

if __name__ == "__main__":

    sample = torch.randn(64, 1, 32, 32)
    model = LeNet(10)
    
    print(model)
    print(model(sample).shape)

