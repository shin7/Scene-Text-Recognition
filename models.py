import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# __all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class CReLUIn(nn.Module):
    """CReLU_IN for the beginning feature extraction based on E2E-MLT model"""

    def __init__(self, channels):
        super(CReLUIn, self).__init__()
        self.bn = nn.InstanceNorm2d(channels * 2, eps=1e-05, momentum=0.1, affine=True)

    def forward(self, x):
        cat = torch.cat((x, -x), 1)
        x = self.bn(cat)
        return F.leaky_relu(x, 0.01, inplace=True)


def _upsample(x, y, scale=1):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=False)


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.shared_feature1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            CReLUIn(16),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            CReLUIn(32)
        )

        self.shared_feature2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer_bn = nn.BatchNorm2d(256)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(256)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(256)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(256)
        self.smooth3_relu = nn.ReLU(inplace=True)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(256)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(256)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(256)
        self.latlayer3_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.f_score = nn.Conv2d(256, 1, (1, 1))
        self.rbox = nn.Conv2d(256, 4, (1, 1))
        self.angle = nn.Conv2d(256, 1, (1, 1))

        # OCR part
        self.ocr_conv5 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1), bias=False)
        self.ocr_conv6 = nn.Conv2d(128, 128, (3, 3), padding=1, bias=False)
        self.ocr_conv7 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.ocr_conv8 = nn.Conv2d(256, 256, (3, 3), padding=1, bias=False)
        self.ocr_conv9 = nn.Conv2d(256, 256, (3, 3), padding=(1, 1), bias=False)
        self.ocr_conv10 = nn.Conv2d(256, 256, (2, 3), padding=(0, 1), bias=False)
        self.ocr_conv11 = nn.Conv2d(256, 7500, (1, 1), padding=(0, 0))

        self.ocr_batch5 = nn.InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.ocr_batch6 = nn.InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.ocr_batch7 = nn.InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.ocr_batch10 = nn.InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.ocr_max2 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.drop1 = nn.Dropout2d(p=0.2, inplace=False)
        self.ocr_leaky = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_ocr(self, x):
        x = self.shared_feature1(x)
        x = self.shared_feature2(x)

        x = self.ocr_conv5(x)
        x = self.ocr_batch5(x)
        x = self.ocr_leaky(x)

        x = self.ocr_conv6(x)
        x = self.ocr_leaky(x)
        x = self.ocr_conv6(x)
        x = self.ocr_leaky(x)

        x = self.ocr_max2(x)
        x = self.ocr_conv7(x)
        x = self.ocr_batch7(x)
        x = self.ocr_leaky(x)

        x = self.ocr_conv8(x)
        x = self.ocr_leaky(x)
        x = self.ocr_conv8(x)
        x = self.ocr_leaky(x)

        x = self.ocr_conv9(x)
        x = self.ocr_leaky(x)
        x = self.ocr_conv9(x)
        x = self.ocr_leaky(x)

        x = self.ocr_max2(x)

        x = self.ocr_conv10(x)
        x = self.ocr_batch10(x)
        x = self.ocr_leaky(x)

        x = self.drop1(x)
        x = self.ocr_conv11(x)
        x = x.squeeze(2)

        x = x.permute(0, 2, 1)
        y = x
        x = x.contiguous().view(-1, x.data.shape[2])
        x = nn.LogSoftmax(len(x.size()) - 1)(x)
        x = x.view_as(y)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        x = self.shared_feature1(x)
        x = self.shared_feature2(x)
        x = self.drop1(x)
        h = x
        # h = self.conv1(h)
        # h = self.bn1(h)
        # h = self.relu1(h)
        # h = self.maxpool(h)

        h = self.layer1(h)
        c2 = h
        h = self.layer2(h)
        c3 = h
        h = self.layer3(h)
        c4 = h
        h = self.layer4(h)
        c5 = h

        # Top-down
        p5 = self.toplayer(c5)
        p5 = self.toplayer_relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)
        c4 = self.latlayer1_relu(self.latlayer1_bn(c4))
        p4 = _upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = self.smooth1_relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = self.latlayer2_relu(self.latlayer2_bn(c3))
        p3 = _upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = self.latlayer3_relu(self.latlayer3_bn(c2))
        p2 = _upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))

        p3 = _upsample(p3, p2)
        p4 = _upsample(p4, p2)
        p5 = _upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), 1)
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))

        f_score = torch.sigmoid(self.f_score(out))
        geo_map = torch.sigmoid(self.rbox(out)) * 512  # text_scale (EAST)
        angle_map = (torch.sigmoid(self.angle(out)) - 0.5) * np.pi / 2  # angle is between [-45, 45]
        f_geometry = torch.cat((geo_map, angle_map), 1)

        return f_score, f_geometry


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in list(state.keys()):
            if key in list(pretrained_model.keys()):
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


if __name__ == '__main__':
    net = resnet50(pretrained=True).cuda()
    score, geo = net(torch.randn(4, 3, 512, 512).cuda())
    print(geo.shape)
