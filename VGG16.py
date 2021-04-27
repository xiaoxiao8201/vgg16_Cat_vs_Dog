import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class my_vgg16(nn.Module):

    def __init__(self, numClass=3):
        super(my_vgg16, self).__init__()
        self.numClass = numClass
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.sf = nn.Softmax(dim=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.fc4096_1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc4096_2 = nn.Linear(4096, 4096)
        self.fc_end = nn.Linear(4096, self.numClass)
        pre_train = torch.load("./vgg16-397923af.pth", map_location=device)
        self._initialize_weights(pre_train)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1(x)
        x = self.conv1_2(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.relu1(x)
        x = self.conv2_2(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv3_1(x)
        x = self.relu1(x)
        x = self.conv3_2(x)
        x = self.relu1(x)
        x = self.conv3_3(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv4_1(x)
        x = self.relu1(x)
        x = self.conv4_2(x)
        x = self.relu1(x)
        x = self.conv4_3(x)
        x = self.relu1(x)

        x = self.maxpool(x)
        x = self.conv5_1(x)
        x = self.relu1(x)
        x = self.conv5_2(x)
        x = self.relu1(x)
        x = self.conv5_3(x)
        x = self.relu1(x)

        x = self.maxpool(x)


        x = x.view(x.size(0), -1)

        x = self.fc4096_1(x)
        x = self.relu1(x)
        x = self.fc4096_2(x)
        x = self.relu1(x)
        x = self.fc_end(x)
        x = self.sf(x)

        return x
#
#     def _initialize_weights(self, pre_train):
#         keys = list(pre_train.keys())
#         self.conv1_1.weight.data.copy_(pre_train[keys[0]])
#         self.conv1_1.bias.data.copy_(pre_train[keys[1]])
#         self.conv1_2.weight.data.copy_(pre_train[keys[2]])
#         self.conv1_2.bias.data.copy_(pre_train[keys[3]])
#
#
#
# w = torch.rand(size=(8, 3, 244, 244))
# vgg = my_vgg16(numClass=2)
# out = vgg(w)
# print(vgg.conv1_1.bias.data)


