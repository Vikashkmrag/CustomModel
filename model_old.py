import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
def conv4x5(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(4,4), 
                     stride=stride, bias=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



# Detection Model With Residual Block
class Detector(nn.Module):
    def __init__(self, block, layers, num_classes=11,max_boxes=5):
        super(Detector, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 1)
        # self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(1024, 128)
        self.bbox = conv4x5(32,4) #nn.Linear(128, max_boxes*4)
        self.clss = conv4x5(32,num_classes) #nn.Linear(128, max_boxes*num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = self.avg_pool(out)
        # print(out.shape)
        # outfc = out.view(out.size(0), -1)
        # outfc = self.fc(outfc)
        # print(out.shape)
        # print("****")
        out_box = self.bbox(out)
        out_clss = self.clss(out)
        out_clss = self.softmax(out_clss)
        # out = self.fc2(out)
        # print(out_box.shape)
        # print(out_clss.shape)
        return out_box,out_clss
    
# model = ResNet(ResidualBlock, [2, 2, 2]).to(device)