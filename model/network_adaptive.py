import torch
import torch.nn as nn
import copy
from model.losses import prototype_loss
from config import device
import torch.nn.functional as F
from config import args

###############################################layer_wise_adaptor####################################################
class conv_adaptive(nn.Module):
    def __init__(self, orig_conv):
        super(conv_adaptive, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        self.film_alpha = nn.Parameter(torch.ones(1, planes))
        self.film_beta = nn.Parameter(torch.zeros(1, planes))
        self.film_alpha.requires_grad = True
        self.film_beta.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        gamma = self.film_alpha.view(1, -1, 1, 1)
        beta = self.film_beta.view(1, -1, 1, 1)
        #y=F.conv2d(x, self.film_alpha, self.film_beta, stride=self.conv.stride)
        y=gamma*y+beta
        return y

class ResNet_MDL_adaptive(nn.Module):
    def __init__(self, orig_resnet):
        super(ResNet_MDL_adaptive, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_adaptive(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_adaptive(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_adaptive(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_adaptive(m)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features

    def forward(self, x):
        return self.backbone.forward(x=x)
    def forward_single(self, x, index):
        return self.backbone.forward_single(x=x, index=index)

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
            if 'film' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
        # initialize pre-classifier alignment mapping (beta)
        #v = self.beta.weight
        #self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)

