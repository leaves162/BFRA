"""
https://blog.csdn.net/weixin_47062350/article/details/109678871
resnet18相关源码，复写，修改MDL相关要用的函数
"""

import torch.nn as nn
import  torch

##################################################basic function###################################################
def conv3x3(input, output, stride=1):
    n3=nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False)
    return n3

def conv1x1(input, output, stride=1):
    n1=nn.Conv2d(input, output, kernel_size=1, stride=stride, bias=False)
    return n1

class BaseNetwork(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1,downsample=None):
        """
        inplanes (int): 输入的Feature Map的通道数
        planes (int): 第一个卷积层输出的Feature Map的通道数
        stride (int, optional): 第一个卷积层的步长
        downsample (nn.Sequential, optional): 旁路下采样的操作
        """
        super(BaseNetwork, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CosineClassifier(nn.Module):
    """
    用于实现resnet中最后一层采用cosine的情况，暂时未实现
    """
    def __init__(self, feature, num_classes):
        super(CosineClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature = feature
###################################################################################################################

################################################basic resnet class##################################################

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=64, dropout=0.0, zero_init_residual=False):
        """
                block (BasicBlock / Bottleneck): 残差块类型
                layers (list): 每一个stage的残差块的数目，长度为4
                               resnet18: 2,2,2,2
                               resnet34: 3,4,6,3
                               resnet50: 3,4,6,3
                               resnet101:3,4,23,3
                               resnet152:3,8,36,3
                num_classes (int): 类别数目
                zero_init_residual (bool): 若为True则将每个残差块的最后一个BN层初始化为零，
                                           这样残差分支从零开始每一个残差分支，每一个残差块表现的就像一个恒等映射，根据
                                           https://arxiv.org/abs/1706.02677这可以将模型的性能提升0.2~0.3%
                """
        super(ResNet, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512
        # handle classifier creation
        if num_classes is not None:
            self.cls_fn = nn.Linear(self.outplanes, num_classes)
        else:
            print('the parameters only build a feature extractor, not a classifier!')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # fan_out随机初始化，使用正态分布
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BaseNetwork):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
                block (BasicBlock / Bottleneck): 残差块结构
                plane (int): 残差块中第一个卷积层的输出通道数
                bloacks (int): 当前Stage中的残差块的数目
                stride (int): 残差块中第一个卷积层的步长
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        x = self.cls_fn(embed)
        return x

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # return x.squeeze()
        return x.flatten(1)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]

###############################################resnet_mdl_class####################################################
class ResNet_MDL(nn.Module):
    def __init__(self, block, layers, num_classes=None,dropout=0.0, zero_init_residual=False):
        super(ResNet_MDL, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512
        self.num_classes = num_classes
        # handle classifiers creation
        if isinstance(num_classes,list):
            cls_fn = []
            for num_class in num_classes:
                cls_fn.append(nn.Linear(self.outplanes, num_class))
            self.cls_fn = nn.ModuleList(cls_fn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BaseNetwork):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_single(self, x, index):
        """
        Args:
            x: 单独使用其中一个cls接口，输出单个域的forward
            index: 第i个域的数据
        """
        x=self.embed(x)
        x=self.cls_fn[index](x)
        return x
    def forward(self, x, num_samples=None, kd=False):
        if kd:#主要就是num_sample分开操作，方便得到各个域上的数据特征
            embed_ = self.embed(x)
            embed = list(torch.split(embed_, num_samples))
            x = self.dropout(torch.cat(embed, dim=0))
        else:
            x = self.embed(x)
            x = self.dropout(x)
        x = list(torch.split(x, num_samples))
        out = []
        for t in range(len(self.num_classes)):
            out.append(self.cls_fn[t](x[t]))
        if kd:
            return out, embed
        else:
            return out

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]

###############################################resnet_aug_class####################################################
class affine_part(nn.Module):
    """softplus noice to play a role of argument."""
    def __init__(self, planes):
        super(affine_part, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, planes))
        self.beta = nn.Parameter(torch.zeros(1, planes))
    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x + beta

class BaseNetwork_affine(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, num_classes=None, downsample=None):
        """
        Args:
            num_samples: None时表示仅有共享的affine层
                         如果是一个列表，那么各个域的数据仅经过自己的affine，另外额外加一个共有层（就是None时的层）
        """
        super(BaseNetwork_affine, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.num_classes=num_classes
        if num_classes==None:
            self.affine1 = affine_part(planes)
            self.affine2 = affine_part(planes)
        elif isinstance(num_classes,list):
            affine1, affine2=[],[]
            for i in range(len(num_classes)+1):
                affine1.append(affine_part(planes))
                affine2.append(affine_part(planes))
            self.affine1=nn.ModuleList(affine1)
            self.affine2=nn.ModuleList(affine2)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.num_classes==None:
            out = self.affine1(out)
        else:
            out_list = list(torch.split(out, self.num_classes))
            out=[]
            for t in range(len(self.num_classes)):
                out.append(self.affine1[t](out_list[t]))
            out.append(self.affine1[-1](torch.cat(out_list, dim=0)))
            out=torch.cat(out, dim=0)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.num_classes == None:
            out = self.affine2(out)
        else:
            out_list = list(torch.split(out, self.num_classes))
            out = []
            for t in range(len(self.num_classes)):
                out.append(self.affine2[t](out_list[t]))
            out.append(self.affine2[-1](torch.cat(out_list, dim=0)))
            out = torch.cat(out, dim=0)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet_MDL_affine(nn.Module):
    def __init__(self, block, layers, num_classes=64,
                 dropout=0.0, zero_init_residual=False, if_static=True):
        super(ResNet_MDL_affine, self).__init__()
        self.initial_pool = False
        self.aug_normalize = affine_part(3)
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512
        self.num_classes = num_classes
        self.if_static=if_static
        # handle classifiers creation
        if isinstance(num_classes, list):
            cls_fn = []
            for num_class in num_classes:
                cls_fn.append(nn.Linear(self.outplanes, num_class))
            self.cls_fn = nn.ModuleList(cls_fn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BaseNetwork_affine):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        temp_num_classes=self.num_classes
        if self.if_static:
            temp_num_classes=None
        layers.append(block(self.inplanes, planes, stride, temp_num_classes, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_classes=temp_num_classes))
        return nn.Sequential(*layers)

    def forward(self, x, num_samples=None, kd=False):
        if kd:#主要就是num_sample分开操作，方便得到各个域上的数据特征
            embed_ = self.embed(x)
            embed = list(torch.split(embed_, num_samples))
            x = self.dropout(torch.cat(embed, dim=0))
        else:
            x = self.embed(x)
            x = self.dropout(x)
        x = list(torch.split(x, num_samples))
        out = []
        for t in range(len(self.num_classes)):
            out.append(self.cls_fn[t](x[t]))
        if kd:
            return out, embed
        else:
            return out

    def embed(self, x):
        x = self.aug_normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.flatten(1)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]

    def get_state_dict_affine(self):
        """Outputs the state elements that are domain-specific"""
        return {k: v for k, v in self.state_dict().items()
                if 'affine' in k or 'cls' in k or 'running' in k}

    def get_parameters_affine(self):
        """Outputs only the parameters that are domain-specific"""
        return [v for k, v in self.named_parameters()
                if 'affine' in k or 'cls' in k]

class BaseNetwork_affine_only(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, num_classes=None, downsample=None):
        """
        Args:
            num_samples: None时表示仅有共享的affine层
                         如果是一个列表，那么各个域的数据仅经过自己的affine，另外额外加一个共有层（就是None时的层）
        """
        super(BaseNetwork_affine_only, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.num_classes=num_classes
        if num_classes==None:
            self.affine1 = affine_part(planes)
            self.affine2 = affine_part(planes)
        elif isinstance(num_classes,list):
            affine1, affine2=[],[]
            for i in range(len(num_classes)+1):
                affine1.append(affine_part(planes))
                affine2.append(affine_part(planes))
            self.affine1=nn.ModuleList(affine1)
            self.affine2=nn.ModuleList(affine2)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.num_classes==None:
            out = self.affine1(out)
        else:
            out_list = list(torch.split(out, self.num_classes))
            out=[]
            for t in range(len(self.num_classes)):
                out.append(self.affine1[t](out_list[t]))
            out.append(self.affine1[-1](torch.cat(out_list, dim=0)))
            out=torch.cat(out, dim=0)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.num_classes == None:
            out = self.affine2(out)
        else:
            out_list = list(torch.split(out, self.num_classes))
            out = []
            for t in range(len(self.num_classes)):
                out.append(self.affine2[t](out_list[t]))
            out.append(self.affine2[-1](torch.cat(out_list, dim=0)))
            out = torch.cat(out, dim=0)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet_MDL_affine(nn.Module):
    def __init__(self, block, layers, num_classes=64,
                 dropout=0.0, zero_init_residual=False, if_static=True):
        super(ResNet_MDL_affine, self).__init__()
        self.initial_pool = False
        self.aug_normalize = affine_part(3)
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512
        self.num_classes = num_classes
        self.if_static=if_static
        # handle classifiers creation
        if isinstance(num_classes, list):
            cls_fn = []
            for num_class in num_classes:
                cls_fn.append(nn.Linear(self.outplanes, num_class))
            self.cls_fn = nn.ModuleList(cls_fn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BaseNetwork_affine):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        temp_num_classes=self.num_classes
        if self.if_static:
            temp_num_classes=None
        layers.append(block(self.inplanes, planes, stride, temp_num_classes, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_classes=temp_num_classes))
        return nn.Sequential(*layers)

    def forward(self, x, num_samples=None, kd=False):
        if kd:#主要就是num_sample分开操作，方便得到各个域上的数据特征
            embed_ = self.embed(x)
            embed = list(torch.split(embed_, num_samples))
            x = self.dropout(torch.cat(embed, dim=0))
        else:
            x = self.embed(x)
            x = self.dropout(x)
        x = list(torch.split(x, num_samples))
        out = []
        for t in range(len(self.num_classes)):
            out.append(self.cls_fn[t](x[t]))
        if kd:
            return out, embed
        else:
            return out

    def embed(self, x):
        x = self.aug_normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.flatten(1)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]

    def get_state_dict_affine(self):
        """Outputs the state elements that are domain-specific"""
        return {k: v for k, v in self.state_dict().items()
                if 'affine' in k or 'cls' in k or 'running' in k}

    def get_parameters_affine(self):
        """Outputs only the parameters that are domain-specific"""
        return [v for k, v in self.named_parameters()
                if 'affine' in k or 'cls' in k]

###############################################resnet_load_function####################################################

def resnet18(pretrained=False, pretrained_model_path=None, **kwargs):

    model = ResNet(BaseNetwork, [2, 2, 2, 2], **kwargs)
    if pretrained and pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {}
        for k, v in ckpt_dict.items():
            if 'cls' not in k:
                shared_state[k]=v
        model.load_state_dict(shared_state, strict=False)#仅同步预训练文件中有的参数，cls这样另加的不初始化
        print('Load pretrained_model: shared weights from {} for ResNet18.'.format(pretrained_model_path))
    else:
        print('Load pretrained_model: No pretrained model, initialize for ResNet18.')
    return model

def resnet18_mdl(pretrained=False, pretrained_model_path=None, **kwargs):
    model = ResNet_MDL(BaseNetwork, [2, 2, 2, 2], **kwargs)
    if pretrained and pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {}
        for k, v in ckpt_dict.items():
            if 'cls' not in k:
                shared_state[k] = v
        model.load_state_dict(shared_state, strict=False)  # 仅同步预训练文件中有的参数，cls这样另加的不初始化
        print('Load pretrained_model: shared weights from {} for ResNet18_MDL.'.format(pretrained_model_path))
    else:
        print('Load pretrained_model: No pretrained model, initialize for ResNet18_MDL.')
    return model


def resnet50(pretrained=True, pretrained_model_path=None, **kwargs):

    model = ResNet(BaseNetwork, [3, 4, 6, 3], **kwargs)
    if pretrained and pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {}
        for k, v in ckpt_dict.items():
            if 'cls' not in k:
                shared_state[k]=v
        model.load_state_dict(shared_state, strict=False)#仅同步预训练文件中有的参数，cls这样另加的不初始化
        print('Load pretrained_model: shared weights from {} for ResNet50.'.format(pretrained_model_path))
    else:
        print('Load pretrained_model: No pretrained model, initialize for ResNet50.')
    return model


def resnet50_mdl(pretrained=True, pretrained_model_path=None, **kwargs):
    model = ResNet_MDL(BaseNetwork, [3, 4, 6, 3], **kwargs)
    if pretrained and pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {}
        for k, v in ckpt_dict.items():
            if 'cls' not in k:
                shared_state[k] = v
        model.load_state_dict(shared_state, strict=False)  # 仅同步预训练文件中有的参数，cls这样另加的不初始化
        print('Load pretrained_model: shared weights from {} for ResNet50_MDL.'.format(pretrained_model_path))
    else:
        print('Load pretrained_model: No pretrained model, initialize for ResNet50_MDL.')
    return model

def resnet18_mdl_aug(pretrained=True, pretrained_model_path=None, **kwargs):
    model = ResNet_MDL_aug(BaseNetwork_aug, [2, 2, 2, 2], **kwargs)
    # loading shared convolutional weights
    if pretrained and pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {}
        for k, v in ckpt_dict.items():
            if 'cls' not in k and 'aug' not in k:
                shared_state[k] = v
        model.load_state_dict(shared_state, strict=False)
        print('Load pretrained_model: shared weights from {} for ResNet50_MDL_aug.'.format(pretrained_model_path))
    else:
        print('Load pretrained_model: No pretrained model, initialize for ResNet50_MDL_aug.')
    return model

###############################################feature transformation class####################################################


class FT(torch.nn.Module):
    def __init__(self, num_datasets, dim_in, dim_out=None, opt='linear'):
        super(FT, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        self.num_datasets = num_datasets

        for i in range(num_datasets):
            if opt == 'linear':
                setattr(self, 'conv{}'.format(i), torch.nn.Conv2d(dim_in, dim_out, 1, bias=False))
            else:
                setattr(self, 'conv{}'.format(i), nn.Sequential(
                    torch.nn.Conv2d(dim_in, 2*dim_in, 1, bias=False),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(2*dim_in, dim_out, 1, bias=False),
                    )
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        for i in range(self.num_datasets):
            ad_layer = getattr(self, 'conv{}'.format(i))
            if len(list(inputs[i].size())) < 4:
                input_ = inputs[i].view(inputs[i].size(0), -1, 1, 1)
            else:
                input_ = inputs[i]
            results.append(ad_layer(input_).flatten(1))
            # results.append(ad_layer(inputs[i]))
        return results

