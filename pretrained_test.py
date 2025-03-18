import os
import argparse
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json

from config import PROJECT_ROOT, TRAIN_METADATASET_NAMES, BATCHSIZES, device, ALL_METADATASET_NAMES
from data.meta_dataset_reader import MetaDatasetBatchReader, MetaDatasetEpisodeReader
from model.base_network import ResNet, BaseNetwork
from model.model_utils import get_initial_optimizer, lr_scheduler, if_contains_nan, calculate_var
from model.model_loss import cross_entropy_loss,prototype_loss, scm_loss

from model.adapt_network import linear_adaptor, skip_adaptor, affine_skip_adaptor
#pretran test: model:url, standard
#              adapt_type:linear_based, skip_based(有affine就是多个affine_skip_based)
#              test_mode:ncc, scm

#test_results命名:{pretrain_model}_{universal_model}_{adaptor}_{test_type}
#                 pretrain_model: url, standard(pretrain_model统一)
#                 universal_model:none_affine, single_affine, multi_affine(model统一)
#                 adaptor:        linear_based, skip_based, affine_skip_based(adapt_model统一)
#                 test_type:      ncc, scm(loss_function统一)

parser = argparse.ArgumentParser(description='adapt test')
parser.add_argument('--dataset_index', type=int, default=0)

parser.add_argument('--pretrain_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train')
parser.add_argument('--pretrain_model_name', type=str, default='url')#url, standard

parser.add_argument('--test_result_output_path', type=str, default=PROJECT_ROOT+'/results')
#所有的test结果输出尽量发在一起，方便对比
parser.add_argument('--test_adapt_model_out_path',test=str, default=PROJECT_ROOT+'/results/meta_test')
parser.add_argument('--adaptor_type', type=str, default='linear_based')
parser.add_argument('--test_size', type=int, default=500)

parser.add_argument('--proto_distance', type=str, default='cos')#l2, cos
parser.add_argument('--dropouut', type=float, default=0.0)

parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--epochs',type=int, default=100)
parser.add_argument('--test_mode', type=str, default='ncc')#ncc, scm
#record
parser.add_argument('--writer_frequency', type=int, default=200)
args = parser.parse_args()
#无affine: 加载ResNet模型，加载ResNet_MDL的参数(cls_fn不加载)
pretrain_model_out_path=args.pretrain_base_path+'/'+args.pretrain_model_name
if args.test_mode=='ncc':
    loss_function=prototype_loss
elif args.test_mode=='scm':
    loss_function=scm_loss

#mean中先放每次的结果，最终mean一下保存均值，方差保存到var中
test_support_acc={name:{'mean':0, 'var':0} for name in ALL_METADATASET_NAMES}
test_target_acc={name:{'mean':0, 'var':0} for name in ALL_METADATASET_NAMES}
writer = SummaryWriter(pretrain_model_out_path)
for t_index, dataset in enumerate(ALL_METADATASET_NAMES):
    trainsets, valsets, testsets = [dataset], [dataset], [dataset]
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args.proto_distance)

    pretrain_model_path = pretrain_model_out_path + '/' + dataset + '/' + dataset + '.pth.tar'
    model = ResNet(BaseNetwork, [2, 2, 2, 2], num_classes=None, dropout=args.dropout)
    if os.path.isfile(pretrain_model_path):
        pretrain_ckpt = torch.load(pretrain_model_path, map_location=device)['state_dict']
        model.load_state_dict(pretrain_ckpt, strict=False)
        model.to(device)
    else:
        raise RuntimeError('there is no pretrain checkpoint in ', pretrain_model_path)
    lr=args.learning_rate
    if dataset not in TRAIN_METADATASET_NAMES:
        lr=lr*10
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        for j in tqdm(range(args.test_size),postfix='pretrain/test/'+dataset):
            with torch.no_grad():
                sample = test_loader.get_test_task(session, dataset)
                context_features = model.embed(sample['context_images'])
                target_features = model.embed(sample['target_images'])
                context_labels = sample['context_labels']
                target_labels = sample['target_labels']
            adapt_model = linear_adaptor(model, context_features.size(1))
            optimizer = torch.optim.Adadelta(adapt_model.transform_layer, lr=lr)#有点疑问
            support_accs, target_accs=[], []
            for test_i in range(args.epochs):
                optimizer.zero_grad()
                adapt_model.forward(context_features)
                support_loss, support_acc = loss_function(context_features, context_labels,
                                                          context_features, context_labels, distance=args.proto_distance)
                support_loss.backward()
                optimizer.step()
            support_accs.append(support_acc.item())
            with torch.no_grad():
                support_feature = adapt_model.forward(context_features)
                target_feature = adapt_model.forward(target_features)
                target_loss, target_acc=loss_function(support_feature, context_labels,
                                                      target_feature, target_labels, distance=args.proto_distance)
            target_accs.append(target_acc.item())
    test_support_acc[dataset]['mean'], test_support_acc[dataset]['var']=calculate_var(support_accs)
    test_target_acc[dataset]['mean'], test_target_acc[dataset]['var']=calculate_var(target_accs)













