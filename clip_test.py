import os
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from tabulate import tabulate

from data.prepare_extra_data import get_bscd_loader

import argparse
import warnings
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import PROJECT_ROOT, TRAIN_METADATASET_NAMES, BATCHSIZES, device, ALL_METADATASET_NAMES
from data.meta_dataset_reader import MetaDatasetBatchReader, MetaDatasetEpisodeReader
from model.base_network import ResNet, BaseNetwork, ResNet_MDL_affine, BaseNetwork_affine
from model.model_utils import get_initial_optimizer, lr_scheduler, if_contains_nan, calculate_var, vote_for_preds,if_contains_zero
from model.model_loss import cross_entropy_loss,prototype_loss, scm_loss, prototype_loss_with_preds

from model.adapt_network import linear_adaptor, skip_adaptor, affine_skip_adaptor

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import model.model_utils as utils
from model.losses import prototype_loss
import CLIP.clip as clip

device='cpu'

templates = [
    'a photo of a {}.',
    'a photo of a small {}.',
    'a photo of a medium {}.',
    'a photo of a large {}.',
    'This is a photo of a {}.',
    'This is a photo of a small {}.',
    'This is a photo of a medium {}.',
    'This is a photo of a large {}.',
    'a {} in the scene.',
    'a photo of a {} in the scene.',
    'There is a {} in the scene.',
    'There is the {} in the scene.',
    'This is a {} in the scene.',
    'This is the {} in the scene.',
    'This is one {} in the scene.',
    ]

def single_template(class_names, model):
    with torch.no_grad():
        texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings

## multi templates
def multi_template(class_names, model, templates):
    with torch.no_grad():
        text_embeddings = []
        for classname in class_names:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) # 多个模板的text
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0) #多个templates的均值embedding
            class_embedding /= class_embedding.norm()
            text_embeddings.append(class_embedding)
        text_embeddings = torch.stack(text_embeddings, dim=0).to(device)
    return text_embeddings

extra_test_dataset=['EuroSAT', 'ISIC', 'CropDisease', 'ChestX']
clip_model, preprocess = clip.load('ViT-B/16', device)

# labels=[['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake'],
#         [],
#         [],
#         []]

for dataset in extra_test_dataset:
    accs = []
    #label_names=labels[0]
    #label_features=multi_template(label_names,clip_model, templates)
    data_loader_test = get_bscd_loader(dataset, 5, 50, 224)
    data_loader_test.generator.manual_seed(3407)
    for data in data_loader_test:
        context_images, context_labels, target_images, target_labels = data
        context_images = context_images.squeeze(0).to(device)
        context_labels = context_labels.squeeze(0).to(device)
        target_images = target_images.squeeze(0).to(device)
        target_labels = target_labels.squeeze(0).to(device)
        #label_set=list(set(target_labels.cpu().numpy()))
        #print(label_set)
        #print('context_images:',context_images.shape)
        #print('context_labels:',context_labels.shape)
        #print('target_images:',target_images.shape)
        #print('target_labels:',target_labels.shape)
        #target_text_embeddings=multi_template(target_labels,clip_model, templates)
        context_image_embedding=clip_model.encode_image(context_images)
        target_image_embeddings=clip_model.encode_image(target_images)
        loss, stats_dict, pred_dict=prototype_loss(context_image_embedding, context_labels,
                                                   target_image_embeddings, target_labels)
        acc=stats_dict['acc']
        accs.append(acc)
        #print('target_image_embeddings:',target_image_embeddings.shape)
        #print('target_text_embeddings:',target_text_embeddings.shape)
        #cos_store=torch.cosine_similarity(target_image_embeddings.unsqueeze(dim=1),label_features.unsqueeze(dim=0),dim=-1)
        #logits=torch.argmax(cos_store, dim=1)
        #print(logits)
        #print(target_labels)
        #preds= torch.eq(logits, target_labels).float().mean()
        #print(preds)
        #accs.append(preds.item())
    print(dataset, np.array(accs).mean())