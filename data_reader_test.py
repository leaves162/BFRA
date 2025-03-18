import os
import sys
import torch
import numpy as np
import tensorflow as tf
from time import sleep

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader,
                                      MetaDatasetEpisodeReader)
from model.losses import cross_entropy_loss, prototype_loss, distillation_loss, DistillKL
from model.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR, WeightAnnealing)
from model.models_dict import DATASET_MODELS_DICT
from model.model_helpers import get_model, get_optimizer, get_domain_extractors
from model.adaptors import adaptor
from config import device
from utils import Accumulator
from config import args, BATCHSIZES, LOSSWEIGHTS, KDFLOSSWEIGHTS, KDPLOSSWEIGHTS, KDANNEALING


trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']

train_loaders = []
num_train_classes = dict()
for t_indx, trainset in enumerate(trainsets):
    train_loaders.append(MetaDatasetBatchReader('train', [trainset], valsets, testsets,batch_size=BATCHSIZES[trainset]))
    num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
print('num_train_classes:')
print(num_train_classes) # {数据集名称: 训练集中的类数量}
#{'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241, 'fungi': 994, 'vgg_flower': 71}
val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)
test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
with tf.compat.v1.Session(config=config) as session:
    samples = []
    images = dict()
    num_samples = []
    # loading images and labels
    print('train_sample:')
    for t_indx, (name, train_loader) in enumerate(zip(trainsets, train_loaders)):
        sample = train_loader.get_train_batch(session)
        #samples.append(sample)
        #{'images','labels','local_classes','dataset_ids','dataset_name'}

        print('images:',sample['images'].shape)#(batch,3,image_size,image_size)
        print(sample['images'])
        print('labels:', sample['labels'].shape)#(batch)
        print(sample['labels'])
        print('local_classes:', sample['local_classes'].shape)#(batch)
        print(sample['local_classes'])
        print('dataset_ids:', sample['dataset_ids'].shape)#(batch) 默认一个batch中都来自于同一个数据集，这里用的batchreader
        print(sample['dataset_ids'])
        print('dataset_name:', sample['dataset_name'])#数据集名称

        print('dataset:',sample['dataset_name'],'  image_num:',sample['images'].shape[0])
        #(448,64,64,64,64,64,64,64)

        #images[name] = sample['images']
        #num_samples.append(sample['images'].size(0))
        #break

    print('val_sample:')
    for valset in valsets:
        sample = val_loader.get_validation_task(session, valset)
        #{'context_images','context_labels','context_gt','target_images','target_labels','target_gt',}

        print('context_images:', sample['context_images'].shape)#(num,3,image_size,image_size)
        print(sample['context_images'])
        print('context_labels:', sample['context_labels'].shape)#(num)内容是0-4
        print(sample['context_labels'])
        print('context_gt:', sample['context_gt'].shape)#(num)内容是0-4对应的真实标签
        print(sample['context_gt'])
        print('target_images:', sample['target_images'].shape)#(num2,3,image_size,image_size)
        print(sample['target_images'])
        print('target_labels:', sample['target_labels'].shape)#(num2)内容是0-4
        print(sample['target_labels'])
        print('target_gt:', sample['target_gt'].shape)#(num2)内容是0-4对应的真实标签
        print(sample['target_gt'])

        print('dataset:',valset,'  context_image_num:', sample['context_labels'].shape[0])
        #num: (5 ,8 ,472,462,291,492,416,442)
        print('dataset:',valset,'  target_image_num:', sample['target_labels'].shape[0])
        #num2:(50,60,90 ,170,70 ,130,65 ,90 )
        #break

    print('test_sample:')
    for testset in testsets:
        sample = test_loader.get_test_task(session, testset)

        print('context_images:', sample['context_images'].shape)  # (num,3,image_size,image_size)
        print(sample['context_images'])
        print('context_labels:', sample['context_labels'].shape)  # (num)内容是0-25
        print(sample['context_labels'])
        print('context_gt:', sample['context_gt'].shape)  # (num)内容是0-25对应的真实标签
        print(sample['context_gt'])
        print('target_images:', sample['target_images'].shape)  # (num2,3,image_size,image_size)
        print(sample['target_images'])
        print('target_labels:', sample['target_labels'].shape)  # (num2)内容是0-25
        print(sample['target_labels'])
        print('target_gt:', sample['target_gt'].shape)  # (num2)内容是0-25对应的真实标签
        print(sample['target_gt'])

        print('dataset:',testset,'  context_image_num:', sample['context_labels'].shape[0])
        #(392,53,204,332,425,427,476,492,492,482)
        print('dataset:',testset,'  target_image_num:', sample['target_labels'].shape[0])
        #(90 ,80,90 ,160,50 ,200,129,160,150,380)
        #break

