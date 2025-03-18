import torch
import gin
import numpy as np

from torch import nn
import torch.nn.functional as F
from model.model_utils import if_contains_nan
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from model.scm import scm
from sklearn.preprocessing import StandardScaler

def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    #if_contains_nan(log_p_y)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    #if_contains_nan(loss)
    acc = torch.eq(preds, labels).float().mean()
    #stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    #print(pred_dict)
    return loss, acc

def cross_entropy_loss_with_preds(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    if_contains_nan(log_p_y)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    if_contains_nan(loss)
    acc = torch.eq(preds, labels).float().mean()
    #stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, acc, pred_dict

def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos'):
    n_way = len(query_labels.unique())
    #print(n_way)
    prots = torch.zeros(n_way, support_embeddings.shape[-1]).type(
        support_embeddings.dtype).to(support_embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = support_embeddings[(support_labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = support_embeddings[(support_labels == i).nonzero(as_tuple=False), :].mean(0)
    prots=prots.unsqueeze(0)#1*cls_num*feature_dim
    #if_contains_nan(prots)
    embeds = query_embeddings.unsqueeze(1)#N*1*feature_dim
    #if_contains_nan(embeds)
    #print(embeds.shape, prots.shape)
    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10
    #if_contains_nan(logits)
    #print(logits.shape)
    #print(logits)
    return cross_entropy_loss(logits, query_labels)

def prototype_loss_with_preds(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos'):
    n_way = len(query_labels.unique())
    prots = torch.zeros(n_way, support_embeddings.shape[-1]).type(
        support_embeddings.dtype).to(support_embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = support_embeddings[(support_labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = support_embeddings[(support_labels == i).nonzero(as_tuple=False), :].mean(0)
    prots=prots.unsqueeze(0)
    if_contains_nan(prots)
    embeds = query_embeddings.unsqueeze(1)
    if_contains_nan(embeds)
    #print(embeds.shape, prots.shape)
    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10
    #if_contains_nan(logits)
    #print(logits.shape)
    #print(logits)
    if torch.isnan(logits).any():
        print('query:',embeds.shape, embeds)
        print('proto:', prots.shape, prots)
        raise RuntimeError('!!!')
    return cross_entropy_loss_with_preds(logits, query_labels)

def transform_labels_to_zero_start(labels):
    label_list=labels.unique().tolist()
    res=torch.zeros(labels.shape).type(
        labels.dtype).to(labels.device)
    for i in range(labels.shape[0]):
        res[i]=label_list.index(labels[i])
    return res

def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots

def feature_difference_loss(stl_feature, mtl_feature, labels):#[batch_size, feature_dim]
    new_labels = transform_labels_to_zero_start(labels)
    n_way = len(new_labels.unique())
    #normal_size = mtl_feature.shape[0]
    # n_way = mtl_feature.shape[0]
    mtl_prots = compute_prototypes(mtl_feature, new_labels, n_way)  # [n_way, feature_dim]
    stl_prots = compute_prototypes(stl_feature, new_labels, n_way)

    mtl_m= torch.mm(mtl_prots, mtl_prots.transpose(0,1))
    stl_m=torch.mm(stl_prots, stl_prots.transpose(0,1))
    f_diff = mtl_m - stl_m
    loss = (f_diff * f_diff).view(-1, 1).sum(0) *1000 / (n_way*n_way)
    return loss

def scm_loss(support_embeddings, support_labels,
            query_embeddings, query_labels, distance='cos', normalize=True):
    n_way = len(query_labels.unique())
    if normalize:
        support_embeddings = F.normalize(support_embeddings, dim=-1, p=2)
        query_embeddings = F.normalize(query_embeddings, dim=-1, p=2)
    logits = torch.logsumexp(scm(support_embeddings, support_labels, query_embeddings), dim=0)
    return cross_entropy_loss(logits, query_labels)

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if_contains_nan(p_s)
        if_contains_nan(p_t)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        if_contains_nan(loss)
        return loss