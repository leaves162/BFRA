import torch
import torch.nn as nn
import copy
from config import device
import torch.nn.functional as F
from model.base_network import ResNet, ResNet_MDL, ResNet_MDL_affine, BaseNetwork, BaseNetwork_affine, affine_part, conv3x3, conv1x1, Bottleneck

#test时基本都是串行的各个域单独测试，所以选择模型时尽量用单头的模型，加载多头模型的参数，忽略掉不要用的头
#无affine: 加载ResNet模型，加载ResNet_MDL的参数(cls_fn不加载)
#有affine: 加载ResNet_MDL_affine单头模型，加载多头参数(affine单头，cls_fn不加载)
##########################################linear adapter###################################################
"""
class linear_adaptor(nn.Module):
    def __init__(self, original_model, planes):
        super(linear_adaptor, self).__init__()
        self.model=original_model
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.transform_plane=planes
        self.transform_inplane=512
        #self.transform_layer=torch.eye(self.transform_plane,  self.transform_inplane)\
         #   .unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)
        self.transform_layer=conv1x1(self.transform_inplane,self.transform_plane)
    def forward(self, x):
        #print('x:',x.shape)
        x=self.model.embed(x)
        #print('modelx:',x.shape)
        x=x.unsqueeze(-1).unsqueeze(-1)
        #x=F.conv2d(x, self.transform_layer)
        x=self.transform_layer(x)
        #print('finalx:',x.shape)
        return x.flatten(1)
"""
class linear_adaptor(nn.Module):
    def __init__(self, feat_dim):
        super(linear_adaptor, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:#just feature
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x
###############################################tsa adapter#############################################
class skip_part(nn.Module):
    def __init__(self, original_conv):
        super(skip_part, self).__init__()
        self.conv = copy.deepcopy(original_conv)
        self.conv.weight.requires_grad=False
        planes, in_planes, _, _ = self.conv.weight.size()
        self.planes=planes
        self.in_planes=in_planes
        self.stride,_=self.conv.stride
        self.skip=nn.Parameter(torch.ones(self.planes, self.in_planes, 1, 1))
        self.skip.requires_grad=True
    def forward(self, x):
        y=self.conv(x)
        y=y+F.conv2d(x, self.skip, stride=self.stride)
        return y

class skip_adaptor(nn.Module):
    def __init__(self, original_model):
        super(skip_adaptor, self).__init__()
        for k, v in original_model.named_parameters():
            v.requires_grad = False

        for block in original_model.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = skip_part(m)
                    setattr(block, name, new_conv)

        for block in original_model.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = skip_part(m)
                    setattr(block, name, new_conv)

        for block in original_model.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = skip_part(m)
                    setattr(block, name, new_conv)

        for block in original_model.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = skip_part(m)
                    setattr(block, name, new_conv)

        self.backbone = original_model
        feat_dim = original_model.layer4[-1].bn2.num_features
        skip_bias = linear_adaptor(feat_dim)
        setattr(self, 'skip_bias', skip_bias)
    def forward(self, x):
        y=self.backbone.embed(x)
        y=self.skip_bias(y)
        return y
    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'skip' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
        v = self.skip_bias.weight
        self.skip_bias.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
#########################################affine skip adaptor#################################################

class affine_skip_part(nn.Module):
    def __init__(self, original_affine):
        super(affine_skip_part, self).__init__()
        self.affine_module = copy.deepcopy(original_affine)
        self.affine_module.affine_gamma.requires_grad=False
        self.affine_module.affine_beta.requires_grad=False
        self.skip1=nn.Parameter(torch.ones(self.affine_module.affine_gamma.shape))
        self.skip2 = nn.Parameter(torch.ones(self.affine_module.affine_beta.shape))
        self.skip1.requires_grad=True
        self.skip2.requires_grad=True
    def forward(self, x):
        #print(self.affine_module.affine_gamma.shape)
        #print(self.skip1.shape)
        gamma = self.affine_module.affine_gamma.view(1, -1, 1, 1)
        #print(gamma.shape)
        gamma=gamma+self.skip1*gamma
        #print(gamma.shape)
        beta=self.affine_module.affine_beta.view(1,-1,1,1)
        beta=beta+self.skip2*beta
        y=gamma*x+beta
        return y
class affine_skip_adaptor(nn.Module):
    def __init__(self, original_model):
        super(affine_skip_adaptor, self).__init__()
        for k, v in original_model.named_parameters():
            v.requires_grad = False

        for block in original_model.layer1:
            for name, m in block.named_children():
                if isinstance(m, affine_part):
                    new_conv = affine_skip_part(m)
                    setattr(block, name, new_conv)

        for block in original_model.layer2:
            for name, m in block.named_children():
                if isinstance(m, affine_part):
                    new_conv = affine_skip_part(m)
                    setattr(block, name, new_conv)

        for block in original_model.layer3:
            for name, m in block.named_children():
                if isinstance(m, affine_part):
                    new_conv = affine_skip_part(m)
                    setattr(block, name, new_conv)

        for block in original_model.layer4:
            for name, m in block.named_children():
                if isinstance(m, affine_part):
                    new_conv = affine_skip_part(m)
                    setattr(block, name, new_conv)
        self.backbone = original_model

    def forward(self, x):
        return self.backbone.single_forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x, None)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'skip' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)*0.0001
                v.requires_grad=True

"""
#这个没有在内部说明grad require情况，用filter在使用时显式说明
class affine_skip_part(affine_part):
    def __init__(self,planes):
        super(affine_skip_part, self).__init__(planes)
        self.skip1=affine_part(planes)
        self.skip2=affine_part(planes)
        self.affine_gamma.requires_grad=False
        self.affine_beta.requires_grad=False
        #总之保证后面所有的网络中只有skip是可动的，其他全部false
    def forward(self, x):
        gammar=self.skip1(self.affine_gamma)
        beta=self.skip2(self.affine_beta)
        x=gammar*x+beta
        return x

class BaseNetwork_affine_skip(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        #Args:
         #   num_samples: None时表示仅有共享的affine层
          #               如果是一个列表，那么各个域的数据仅经过自己的affine，另外额外加一个共有层（就是None时的层）
        super(BaseNetwork_affine_skip, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.affine1 = affine_skip_part(planes)
        self.affine2 = affine_skip_part(planes)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.affine1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.affine2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class affine_skip_adaptor(nn.Module):
    def __init__(self, block, layers, num_classes=64,
                 dropout=0.0, zero_init_residual=False):
        super(affine_skip_adaptor, self).__init__()
        self.initial_pool = False
        self.affine_normalize = affine_skip_part(3)
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
        if num_classes is not None:
            self.cls_fn=nn.Linear(self.outplanes, self.num_class)

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
                elif isinstance(m, BaseNetwork_affine_skip):
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
    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        x = self.cls_fn(embed)
        return x

    def embed(self, x):
        x = self.affine_normalize(x)
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
        #Outputs all the state elements
        return self.state_dict()

    def get_parameters(self):
        #Outputs all the parameters
        return [v for k, v in self.named_parameters()]

    def get_state_dict_affine(self):
        #Outputs the state elements that are domain-specific
        return {k: v for k, v in self.state_dict().items()
                if 'affine' in k or 'cls' in k or 'running' in k}

    def get_parameters_affine(self):
        #Outputs only the parameters that are domain-specific
        return [v for k, v in self.named_parameters()
                if 'affine' in k or 'cls' in k]
"""


