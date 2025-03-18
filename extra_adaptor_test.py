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
#extra_test_dataset=['EuroSAT', 'ISIC', 'CropDisease', 'ChestX']
extra_test_dataset=['ChestX']

parser = argparse.ArgumentParser(description='adapt test')
parser.add_argument('--dataset_index', type=int, default=0)

parser.add_argument('--pretrain_part', type=bool, default=True, help='基础pretrain参数，基本都是True')
parser.add_argument('--pretrain_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/pretrained_model')
parser.add_argument('--pretrain_model_name', type=str, default='url')#url, standard

parser.add_argument('--universal_part', type=bool, default=True, help='false时affine无效，adapt只能选择linear和skip')
parser.add_argument('--universal_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/universal_part')
parser.add_argument('--universal_model_name', type=str, default='feature_diff/with_affine_universal/med1')#standard, feature_diff
parser.add_argument('--universal_model', type=str, default='universal_part_checkpoint.pth.tar')

parser.add_argument('--affine_part',type=bool,default=True, help='false时是基础MDL，True时分静态和动态')
parser.add_argument('--affine_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/affine_part')
parser.add_argument('--affine_model_name', type=str, default='dynamic_universal/med1')#static_universal, dynamic_universal

parser.add_argument('--adaptor_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_test')
parser.add_argument('--adaptor_model_name', type=str, default='full_based')#linear_based, skip_based, affine_skip_based, none, #partial_based, front_based, full_based
parser.add_argument('--test_type', type=str, default='ncc')#ncc, scm

#parser.add_argument('--test_result_output_path', type=str, default=PROJECT_ROOT+'/results')
#所有的test结果输出尽量发在一起，方便对比
parser.add_argument('--supp_query_num',type=int, default=10)
parser.add_argument('--support_num', type=int, default=10)
parser.add_argument('--test_size', type=int, default=1)
parser.add_argument('--proto_distance', type=str, default='cos')#l2, cos
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=1.0)
parser.add_argument('--epochs',type=int, default=100)
parser.add_argument('--writer_frequency', type=int, default=200)
args = parser.parse_args()
print('extra_data_set_test:')
print('adaptor_model_name:',args.adaptor_model_name)
print('sepport_num:',args.support_num)
print('universal model:',args.universal_model)
#参数检验
if args.universal_part==False:
    if args.affine_part==True:
        raise RuntimeError('when universal_part is False, affine_part can not be True!')
    if args.adaptor_model_name=='affine_skip_based':
        raise RuntimeError('when universal_part is False, adaptor can not select affine_skip_based mode!')
if args.affine_part==False:
    if args.adaptor_model_name=='affine_skip_based':
        raise RuntimeError('when affine_part is False, adaptor can not select affine_skip_based mode!')

pretrain_model_path=args.pretrain_model_base_path+'/'+args.pretrain_model_name    #后面+/dataset/dataset.pth.tar
universal_model_path=args.universal_model_base_path+'/'+args.universal_model_name #后面+/universal_part.pth.tar
affine_model_path=args.affine_model_base_path+'/'+args.affine_model_name          #后面+/multi_head_affine.pth.tar
adaptor_model_path=args.adaptor_model_base_path+'/'+args.adaptor_model_name       #后面加模型名，保存support训练的adaptor参数
adaptor_result_output_path='/sda1/st_data/code/TP_FDAS/results/meta_test/extra_data'
if args.test_type=='ncc':
    loss_function=prototype_loss
elif args.test_type=='scm':
    loss_function=scm_loss

#mean中先放每次的结果，最终mean一下保存均值，方差保存到var中
test_support_acc={name:{'mean':0, 'var':0} for name in extra_test_dataset}
test_target_acc={name:{'mean':0, 'var':0} for name in extra_test_dataset}
#writer = SummaryWriter(args.test_result_output_path)

#把所有参数禁止的模型部分都加载出来
if args.affine_part==False:
    base_model=ResNet(BaseNetwork, [2,2,2,2], num_classes=None, dropout=args.dropout)
else:
    base_model=ResNet_MDL_affine(BaseNetwork_affine, [2,2,2,2], num_classes=None, dropout=args.dropout)
if args.universal_part==True:
    if os.path.isfile(universal_model_path+'/'+args.universal_model):
        load_universal_model=torch.load(universal_model_path+'/'+args.universal_model, map_location=device)['state_dict']
        base_model.load_state_dict(load_universal_model, strict=False)
        base_model.to(device)
    else:
        raise RuntimeError('there is no universal model parameters in ', universal_model_path)
if args.affine_part:
    if os.path.isfile(affine_model_path+'/multi_head_affine.pth.tar'):
        seen_load_affine_model=torch.load(affine_model_path+'/multi_head_affine.pth.tar', map_location=device)['affine']
        #seen_affine_part_params={k:v for k,v in seen_load_affine_model if 'affine' in k}
        #base_model.load_state_dict(load_affine_model, strict=False)
        #base_model.to(device)
    else:
        raise RuntimeError('there is no multi_head_affine model parameters in ', affine_model_path)
    if os.path.isfile(affine_model_path+'/single_head_affine_checkpoint.pth.tar'):
        unseen_load_affine_model = torch.load(affine_model_path + '/single_head_affine_checkpoint.pth.tar', map_location=device)['affine']
    else:
        raise RuntimeError('there is no single_head_affine model parameters in ', affine_model_path)

base_model.load_state_dict(unseen_load_affine_model, strict=False)
base_model.to(device)
for k, v in base_model.named_parameters():
    v.requires_grad = False

adapt_model = base_model
lr = args.learning_rate
seed=1010
print(args.epochs, args.learning_rate)
#base_model=nn.DataParallel(base_model)'
supp_query_num=args.supp_query_num
testsets=extra_test_dataset
for dataset in testsets:
    #print(dataset)
    data_loader_test=get_bscd_loader(dataset, 5, args.support_num+supp_query_num, 84)
    data_loader_test.generator.manual_seed(seed+10000)
    #test_loader=data_loader_test.iterable
    #先model后重复迭代
    #一个数据集上的一个模型，对testloader循环
    #一个数据集上的多个模型的性能平均，testloader循环，testsize循环
    #多个数据集上的各自性能，再加上一个对testsets的循环
    support_accs, target_accs = [], []
    for j in tqdm(range(args.test_size), postfix='adaptor/train_test/' + dataset):
        temp_support_accs, temp_target_accs=[],[]
        #print('test_size ',j)
        mm=0
        for data in data_loader_test:
            #print('test_loader:',mm)
            mm+=1
            context_features, context_labels, target_features, target_labels=data
            context_features=context_features.squeeze(0).to(device)
            context_labels=context_labels.squeeze(0).to(device)
            target_features=target_features.squeeze(0).to(device)
            target_labels=target_labels.squeeze(0).to(device)
            if args.adaptor_model_name in ['partial_based','front_based','full_based']:
                adapt_model.load_state_dict(unseen_load_affine_model,strict=False)
                adapt_model.train()
                adapt_model.to(device)
            else:
                raise RuntimeError('In extra_dataset test, model adaptor can only choice affine!')
            if args.adaptor_model_name == 'partial_based':
                layer_name = 'affine2'
            elif args.adaptor_model_name == 'front_based':
                layer_name = 'affine1'
            elif args.adaptor_model_name == 'full_based':
                layer_name = 'affine'
            for k, v in adapt_model.named_parameters():
                if layer_name in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = False
            optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, adapt_model.parameters()),
                                             lr=lr)

            for test_i in range(args.epochs):
                #print('epochs:',test_i)
                optimizer.zero_grad()
                context_adapt_features = adapt_model.single_forward(context_features)
                #context_adapt_features=torch.reshape(context_adapt_features, (5,args.support_num+supp_query_num,512))
                #temp_context_label=torch.reshape(context_labels,(5,args.support_num+supp_query_num))
                #temp_support_feature=context_adapt_features[:,:args.support_num,:].reshape(1,-1,512).squeeze(0)
                #temp_target_feature=context_adapt_features[:,-supp_query_num:,:].reshape(1,-1,512).squeeze(0)
                #temp_support_labels=temp_context_label[:,:args.support_num].reshape(1,-1).squeeze(0)
                #temp_target_labels=temp_context_label[:,-supp_query_num:].reshape(1,-1).squeeze(0)
                #print('temp_support:',temp_support_feature.shape)
                #print('temp_query:',temp_target_feature.shape)
                #target_feature = adapt_model.single_forward(target_features)
                #if_contains_nan(context_adapt_features)

                #print('feature:',context_adapt_features)
                #print('labels:',context_labels)
                #print(temp_support_labels)
                #print(temp_target_labels)
                #print(target_labels)
                #support_loss, support_acc = loss_function(temp_support_feature, temp_support_labels,
                 #                                         temp_target_feature, temp_target_labels,
                  #                                      distance=args.proto_distance)
                support_loss, support_acc = loss_function(context_adapt_features, context_labels,
                                                          context_adapt_features, context_labels,
                                                          distance=args.proto_distance)
                #support_loss, support_acc = cross_entropy_loss(context_adapt_features, context_labels)
                #print(test_i,' acc:',support_acc.item())
                #print(test_i, ' loss:',support_loss.item())
                if torch.isnan(support_loss).any() or torch.isnan(support_acc).any():
                    print('train feature has nan')
                    print('params:')
                    for k, v in adapt_model.named_parameters():
                        if layer_name in k:
                            print(k, v)
                    print('features:', context_adapt_features)
                    break
                support_loss.backward()
                optimizer.step()
                """
                print('after training:', test_i)
                for k, v in base_model.named_parameters():
                    print(k, v)
                    break
                for k, v in base_model.named_parameters():
                    if 'affine1' in k:
                        print(k, v)
                        break
                """
            with torch.no_grad():
                support_feature = adapt_model.single_forward(context_features)
                target_feature = adapt_model.single_forward(target_features)
                #print(support_feature.shape)
                #rint(target_feature.shape)
                target_loss, target_acc = loss_function(support_feature, context_labels,
                                                        target_feature, target_labels,
                                                        distance=args.proto_distance)
                #target_loss, target_acc=cross_entropy_loss(target_feature, target_labels)
                #print('target:',target_acc.item())
                if torch.isnan(target_loss).any() or torch.isnan(target_acc).any():
                    print('test feature has nan')
                    print('params:')
                    for k, v in adapt_model.named_parameters():
                        if layer_name in k:
                            print(k, v)
                    print('target features:', target_feature)
                    print('raw data:', target_features)
                    break
            temp_support_accs.append(support_acc.item())
            temp_target_accs.append(target_acc.item())
        #print('support:',np.array(temp_support_accs).mean())
        #print('query:',np.array(temp_target_accs).mean())
        print('data splits:',mm)
        support_accs.append(np.array(temp_support_accs).mean())
        target_accs.append(np.array(temp_target_accs).mean())
        #print('after training:')
        #for k, v in base_model.named_parameters():
         #   print(k, v)
          #  break
        #for k, v in base_model.named_parameters():
         #   if 'affine1' in k:
          #      print(k, v)
           #     break
    tag_name = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
    b = [0] * 10
    for i in range(len(target_accs)):
        temp = int(target_accs[i] * 10)
        if temp >= 9:
            temp = 9
        elif temp <= 0:
            temp = 0
        b[temp] += 1
    plt.bar(range(len(tag_name)), b, tick_label=tag_name)
    plt.title(dataset)
    plt.savefig(adaptor_result_output_path + '/' + dataset + '_' + str(args.epochs)  + '_add_medmnist.jpg')
    plt.close()
    test_support_acc[dataset]['mean'], test_support_acc[dataset]['var'] = calculate_var(support_accs)
    test_target_acc[dataset]['mean'], test_target_acc[dataset]['var'] = calculate_var(target_accs)
    print(dataset, ':')
    print('train_acc: ', test_support_acc[dataset]['mean'], '+-', test_support_acc[dataset]['var'])
    print('test_acc: ', test_target_acc[dataset]['mean'], '+-', test_target_acc[dataset]['var'])

pretrain_info='extra_data'
if args.universal_part==True:
    universal_info=args.universal_model_name.split('/')[0]
else:
    universal_info='None'
if args.affine_part==True:
    affine_info=args.affine_model_name
else:
    affine_info='None'
adaptor_info=args.adaptor_model_name
test_type=args.test_type

final_info={'pretrain':pretrain_info,
            'universal':universal_info,
            'affine':affine_info,
            'adaptor':adaptor_info,
            'test_type':test_type,
            'epochs':args.epochs}
final_data={'train':test_support_acc, 'test':test_target_acc}
final_info.update(final_data)
final_info_json=json.dumps(final_info)
test_result_output_name=adaptor_result_output_path+'/'+pretrain_info+'_'\
                        +universal_info+'_'+affine_info+'_'+adaptor_info+'_'+test_type+'_'+args.universal_model+'_'+str(args.epochs)+'_'+str(args.learning_rate)+'_medmnist.json'
f=open(test_result_output_name, 'w',encoding='utf8')
f.write(final_info_json)
f.close()
