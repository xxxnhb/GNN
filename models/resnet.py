import torch.nn as nn

class ResNet12Block(nn.Module):
    """
    ResNet Block
    """
    def __init__(self, inplanes, planes):
        super(ResNet12Block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out

class ResNet(nn.Module):
    def __init__(self, emb_size, block=ResNet12Block):
        super(ResNet, self).__init__()
        cfg = [64, 128, 256, 512]
        iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU()
        self.emb_size = emb_size
        self.layer1 = self._make_layer(block, cfg[0], cfg[0])
        self.layer2 = self._make_layer(block, cfg[0], cfg[1])
        self.layer3 = self._make_layer(block, cfg[1], cfg[2])
        self.layer4 = self._make_layer(block, cfg[2], cfg[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes):
        layers = []
        layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 3 -> 64
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 64 -> 64
        out = self.layer1(out)
        # 64 -> 128
        out = self.layer2(out)
        # 128 -> 256
        out = self.layer3(out)
        # 256 -> 512
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # 512 -> 128
        out = self.layer_last(out)
        return out

def resnet12(emb_size=128):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(emb_size=emb_size)
    return model