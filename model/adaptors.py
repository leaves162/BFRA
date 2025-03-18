import torch
import torch.nn as nn
import copy
from model.losses import prototype_loss
from config import device
import torch.nn.functional as F
from model.network_adaptive import conv_adaptive
###############################################adaptor####################################################
def feature_transform(features, vartheta):
    features = features.unsqueeze(-1).unsqueeze(-1)
    features = F.conv2d(features, vartheta[0]).flatten(1)
    return features

def adaptor(context_features, context_labels, max_iter=50, ad_opt='linear', lr=0.01, distance='l2'):

    input_dim = context_features.size(1)
    output_dim = input_dim
    vartheta = []
    if ad_opt == 'linear':
        #手动定义1*1卷积核作为tranform矩阵
        vartheta.append(torch.eye(output_dim, input_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True))
    optimizer = torch.optim.Adadelta(vartheta, lr=lr)
    for i in range(max_iter):
        optimizer.zero_grad()
        features = context_features.unsqueeze(-1).unsqueeze(-1)
        features = F.conv2d(features, vartheta[0]).flatten(1)
        loss, stat, _ = prototype_loss(features, context_labels,
                                       features, context_labels, distance=distance)
        loss.backward()
        optimizer.step()
    return vartheta

###############################################layer_wise_adaptor####################################################

class tsa_conv(nn.Module):
    def __init__(self, orig_conv):
        super(tsa_conv, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        #if 'alpha' in args['test.tsa_opt']:
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        #if 'alpha' in args['test.tsa_opt']:
        # residual adaptation in matrix form
        y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        return y

class original_adaptor(nn.Module):
    def __init__(self, feat_dim):
        super(original_adaptor, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:#just feature
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x

class tsa_adaptor(nn.Module):
    def __init__(self, orig_resnet):
        super(tsa_adaptor, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = tsa_conv(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = tsa_conv(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = tsa_conv(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = tsa_conv(m)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = original_adaptor(feat_dim)
        setattr(self, 'beta', beta)

    def forward(self, x):
        return self.backbone.forward(x=x)

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
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001

        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)

class conv_adaptive_test(nn.Module):
    def __init__(self, orig_conv):
        super(conv_adaptive_test, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.conv.weight.requires_grad = False
        self.conv.film_alpha.requires_grad = False
        self.conv.film_beta.requires_grad = False
        planes, in_planes, _, _ = self.conv.conv.weight.size()
        stride, _ = self.conv.conv.stride
        self.tsa_alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        #self.tsa_beta = nn.Parameter(torch.zeros(1, planes))
        self.tsa_alpha.requires_grad = True
        #self.tsa_beta.requires_grad = True

    def forward(self, x):
        y = self.conv.conv(x)
        z=self.conv.film_alpha*y+self.conv.film_beta
        #gamma = self.tsa_alpha.view(1, -1, 1, 1)
        #beta = self.tsa_beta.view(1, -1, 1, 1)
        #z=y+F.conv2d(x, self.tsa_alpha, stride=self.conv.conv.stride)
        z=z+y
        return z

class adapive_adaptor(nn.Module):
    def __init__(self, orig_resnet):
        super(adapive_adaptor, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.backbone.layer1:
            for name, m in block.named_children():
                if isinstance(m, conv_adaptive) and m.conv.kernel_size[0] == 3:
                    new_conv = conv_adaptive_test(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.backbone.layer2:
            for name, m in block.named_children():
                if isinstance(m, conv_adaptive) and m.conv.kernel_size[0] == 3:
                    new_conv = conv_adaptive_test(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.backbone.layer3:
            for name, m in block.named_children():
                if isinstance(m, conv_adaptive) and m.conv.kernel_size[0] == 3:
                    new_conv = conv_adaptive_test(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.backbone.layer4:
            for name, m in block.named_children():
                if isinstance(m, conv_adaptive) and m.conv.kernel_size[0] == 3:
                    new_conv = conv_adaptive_test(m)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.backbone.layer4[-1].bn2.num_features
        beta = original_adaptor(feat_dim)
        setattr(self, 'se_beta', beta)

    def forward(self, x):
        return self.backbone.forward(x=x)

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
            if 'tsa_alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001

        # initialize pre-classifier alignment mapping (beta)
        #v = self.beta.weight
        #self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)

def layer_wise_adaptor(context_images, context_labels, model, max_iter=40, lr=0.1, lr_beta=1, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone,
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model=model.to(device)
    model.eval()
    #tsa_opt = args['test.tsa_opt']
    ##########################################################################
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    params = []
    #if 'alpha' in tsa_opt:
    params.append({'params': alpha_params})
    #if 'beta' in tsa_opt:
    params.append({'params': beta_params, 'lr': lr_beta})

    optimizer = torch.optim.Adadelta(params, lr=lr)

    #if 'alpha' not in tsa_opt:
        #with torch.no_grad():
            #context_features = model.embed(context_images)
    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        #if 'alpha' in tsa_opt:
        # adapt features by task-specific adapters
        context_features = model.embed(context_images)
        #if 'beta' in tsa_opt:
            # adapt feature by PA (beta)
        aligned_features = model.beta(context_features)
        #else:
            #aligned_features = context_features
        loss, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)
        loss.backward()
        optimizer.step()
    return

def layer_wise_adaptor_adaptive(context_images, context_labels, model, max_iter=40, lr=0.01, lr_beta=1, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone,
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model=model.to(device)
    model.eval()
    #tsa_opt = args['test.tsa_opt']
    ##########################################################################
    alpha_params = [v for k, v in model.named_parameters() if 'tsa_alpha' in k]
    beta_params = [v for k, v in model.named_parameters() if 'tsa_beta' in k]
    params = []
    #if 'alpha' in tsa_opt:
    params.append({'params': alpha_params})
    #if 'beta' in tsa_opt:
    params.append({'params': beta_params, 'lr': lr_beta})

    optimizer = torch.optim.Adadelta(params, lr=lr)

    #if 'alpha' not in tsa_opt:
        #with torch.no_grad():
            #context_features = model.embed(context_images)
    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        #if 'alpha' in tsa_opt:
        # adapt features by task-specific adapters
        aligned_features = model.embed(context_images)
        #if 'beta' in tsa_opt:
            # adapt feature by PA (beta)
        aligned_features = model.se_beta(aligned_features)
        #else:
            #aligned_features = context_features
        loss, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)
        loss.backward()
        optimizer.step()
    return