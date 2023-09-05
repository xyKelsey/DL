from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, in_channel, out_channel, stride=1, extra=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.extra = extra

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = x #虚线输出值
        if self.extra is not None:
            residual = self.extra(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet18网络
    _make_layer函数实现残差块的重复
    """

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),    #7x7,64,stride 2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)) #output 112x112x64

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #output 56x56x64

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 64, 2) #conv2_x 56x56x64
        self.layer2 = self._make_layer(64, 128, 2, stride=2) #conv3_x
        self.layer3 = self._make_layer(128, 256, 2, stride=2) #conv4_x
        self.layer4 = self._make_layer(256, 512, 2, stride=2) #conv5_x

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 2)


    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        extra = None
        if stride != 1 :
            extra = nn.Sequential( #右边分支1x1卷积层
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

        layers = []
        layers.append(ResidualBlock(in_channel, out_channel, stride, extra))

        for _ in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x)
        x = self.fc(x)
        return x