import os
import argparse
import warnings
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from config import PROJECT_ROOT, TRAIN_METADATASET_NAMES, BATCHSIZES, device, ALL_METADATASET_NAMES
from data.meta_dataset_reader import MetaDatasetBatchReader, MetaDatasetEpisodeReader
from model.base_network import ResNet, BaseNetwork, ResNet_MDL_affine, BaseNetwork_affine
from model.model_utils import get_initial_optimizer, lr_scheduler, if_contains_nan, calculate_var, vote_for_preds,if_contains_zero
from model.model_loss import cross_entropy_loss,prototype_loss, scm_loss, prototype_loss_with_preds

from model.adapt_network import linear_adaptor, skip_adaptor, affine_skip_adaptor
#pretran test: model:url, standard
#              adapt_type:linear_based, skip_based(有affine就是多个affine_skip_based)
#              test_mode:ncc, scm

#test_results命名:{pretrain_model}_{universal_model}_{adaptor}_{test_type}
#                 pretrain_model: url, standard(pretrain_model统一)
#                 universal_model:none_affine, single_affine, multi_affine(model统一)
#                 adaptor:        linear_based, skip_based, affine_skip_based(adapt_model统一)
#                 test_type:      ncc, scm(loss_function统一)
host_datasets=[ALL_METADATASET_NAMES,
               ['ilsvrc_2012','omniglot','aircraft','cu_birds','dtd','quickdraw','fungi','vgg_flower']]

parser = argparse.ArgumentParser(description='adapt test')
parser.add_argument('--dataset_index', type=int, default=0)
parser.add_argument('--test_dataset_index', type=int, default=0)
parser.add_argument('--pretrain_part', type=bool, default=True, help='基础pretrain参数，基本都是True')
parser.add_argument('--pretrain_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/pretrained_model')
parser.add_argument('--pretrain_model_name', type=str, default='url')#url, standard

parser.add_argument('--universal_part', type=bool, default=False, help='false时affine无效，adapt只能选择linear和skip')
parser.add_argument('--universal_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/universal_part')
parser.add_argument('--universal_model_name', type=str, default='standard')#standard, feature_diff

parser.add_argument('--affine_part',type=bool,default=False, help='false时是基础MDL，True时分静态和动态')
parser.add_argument('--affine_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/affine_part')
parser.add_argument('--affine_model_name', type=str, default='dynamic_universal')#static_universal, dynamic_universal

parser.add_argument('--adaptor_model_base_path', type=str, default=PROJECT_ROOT+'/results/meta_test')
parser.add_argument('--adaptor_model_name', type=str, default='none')#linear_based, skip_based, affine_skip_based, none, #partial_based, front_based, full_based
parser.add_argument('--test_type', type=str, default='ncc')#ncc, scm

#parser.add_argument('--test_result_output_path', type=str, default=PROJECT_ROOT+'/results')
#所有的test结果输出尽量发在一起，方便对比

parser.add_argument('--datasets',type=list, default=ALL_METADATASET_NAMES)
parser.add_argument('--test_size', type=int, default=500)
parser.add_argument('--proto_distance', type=str, default='cos')#l2, cos
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=0.5)
parser.add_argument('--epochs',type=int, default=10)
parser.add_argument('--writer_frequency', type=int, default=200)
args = parser.parse_args()
print('epochs:', args.epochs)
print('adaptor_model_name:',args.adaptor_model_name)
train_valid_avg_loss_threshold=[2.18,0.27,0.65, 1.11, 0.79, 1.26, 1.72, 0.71]
unseen_loss_threshold=max(train_valid_avg_loss_threshold)
train_valid_avg_loss_threshold.append(unseen_loss_threshold)
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
adaptor_result_output_path=adaptor_model_path
if args.test_type=='ncc':
    loss_function=prototype_loss
elif args.test_type=='scm':
    loss_function=scm_loss

#mean中先放每次的结果，最终mean一下保存均值，方差保存到var中
test_support_acc={name:{'mean':0, 'var':0} for name in ALL_METADATASET_NAMES}
test_target_acc={name:{'mean':0, 'var':0} for name in ALL_METADATASET_NAMES}
#writer = SummaryWriter(args.test_result_output_path)

#把所有参数禁止的模型部分都加载出来
if args.affine_part==False:
    base_model=ResNet(BaseNetwork, [2,2,2,2], num_classes=None, dropout=args.dropout)
else:
    base_model=ResNet_MDL_affine(BaseNetwork_affine, [2,2,2,2], num_classes=None, dropout=args.dropout)
if args.universal_part==True:
    if os.path.isfile(universal_model_path+'/universal_part.pth.tar'):
        load_universal_model=torch.load(universal_model_path+'/universal_part.pth.tar', map_location=device)['state_dict']
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
    if os.path.isfile(affine_model_path+'/single_head_affine.pth.tar'):
        unseen_load_affine_model = torch.load(affine_model_path + '/single_head_affine.pth.tar', map_location=device)['affine']
    else:
        raise RuntimeError('there is no single_head_affine model parameters in ', affine_model_path)


trainsets, valsets, testsets = TRAIN_METADATASET_NAMES, TRAIN_METADATASET_NAMES,ALL_METADATASET_NAMES
aa=["aircraft","cu_birds","dtd","vgg_flower","traffic_sign","mnist"]
a=['ilsvrc_2012','omniglot','quickdraw','fungi','mscoco','cifar10','cifar100']

print(testsets)
test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type='standard')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.inter_op_parallelism_threads = 1  # 控制 TensorFlow 进程在执行跨操作并发时使用的线程数
#config.intra_op_parallelism_threads = 1  # 控制 TensorFlow 进程在执行单个操作时使用的线程数

tensor_name='meta_train model path'
train_feature_mean=torch.load(tensor_name+'/train_proto.pth',map_location=device)
train_feature_norm=torch.load(tensor_name+'/train_var.pth',map_location=device)

with tf.compat.v1.Session(config=config) as session:
    for dataset in testsets:
        if args.universal_part==False:
            if os.path.isfile(pretrain_model_path+'/'+dataset+'/'+dataset+'.pth.tar'):
                load_pretrain_model = torch.load(pretrain_model_path+'/'+dataset+'/'+dataset+'.pth.tar', map_location=device)['state_dict']
                base_model.load_state_dict(load_pretrain_model, strict=False)
                base_model.to(device)
            else:
                raise RuntimeError('there is no pretrain model parameters in ', pretrain_model_path)
        if args.affine_part:
            if dataset in TRAIN_METADATASET_NAMES:
                test_index=TRAIN_METADATASET_NAMES.index(dataset)
            #else:
                #test_index=len(TRAIN_METADATASET_NAMES)
                #如果是未知域，就用模型中多余的那个head参数来加载
                seen_load_affine_model_params={}
                seen_load_affine_model_params['affine_normalize.affine_gamma']=seen_load_affine_model['affine_normalize.affine_gamma']
                seen_load_affine_model_params['affine_normalize.affine_beta']=seen_load_affine_model['affine_normalize.affine_beta']
                for k, v in seen_load_affine_model.items():
                    if k[17] == str(test_index):
                        temp_k = k[:17] + k[19:]
                        seen_load_affine_model_params[temp_k] = v
                #print(seen_load_affine_model_params.keys())
                base_model.load_state_dict(seen_load_affine_model_params, strict=False)
                base_model.to(device)
            else:
                base_model.load_state_dict(unseen_load_affine_model, strict=False)
                base_model.to(device)
        for k, v in base_model.named_parameters():
            v.requires_grad=False
        base_model.eval()
        lr=args.learning_rate
        #if dataset not in TRAIN_METADATASET_NAMES:
         #   lr=lr*10
        support_accs, target_accs = [], []
        for j in tqdm(range(args.test_size),postfix='adaptor/train_test/'+dataset):
            continue_flag=0
            with torch.no_grad():
                sample = test_loader.get_test_task(session, dataset)
                context_features = sample['context_images']#batch*3*84*84
                target_features = sample['target_images']#batch
                context_labels = sample['context_labels']
                target_labels = sample['target_labels']
            if j==args.test_size-1:
                support_feature = base_model.embed(context_features)
                target_feature = base_model.embed(target_features)
                all_test_feature = torch.cat([support_feature, target_feature],dim=0)
                test_feature_mean=torch.mean(all_test_feature, dim=0)
                test_feature_norm=torch.norm(torch.var(all_test_feature,dim=0), p=2, dim=0)
                dis=torch.norm(train_feature_mean-test_feature_mean, p=2, dim=0)
                print(dataset,'before train:dis:',dis.item(),' train_norm:', train_feature_norm,' test_norm:',test_feature_norm)
            #with torch.no_grad():
             #   context_features=base_model.embed(context_features)
              #  target_features=base_model.embed(target_features)
            if args.adaptor_model_name=='none':
                support_feature = base_model.embed(context_features)
                target_feature = base_model.embed(target_features)
                target_loss, target_acc = loss_function(support_feature, context_labels,
                                                        target_feature, target_labels, distance=args.proto_distance)
                target_accs.append(target_acc.item())
            elif args.adaptor_model_name=='linear_based':
                context_features = base_model.embed(context_features)
                target_features = base_model.embed(target_features)
                vartheta = []
                feature_dim = context_features.size(1)
                vartheta.append(
                    torch.eye(feature_dim, feature_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True))
                #print('we need train a single conv2d layer')
                optimizer = torch.optim.Adadelta(vartheta, lr=lr)
                for i in range(args.epochs):
                    optimizer.zero_grad()
                    features = context_features.unsqueeze(-1).unsqueeze(-1)
                    features = F.conv2d(features, vartheta[0]).flatten(1)
                    support_loss, support_acc = prototype_loss(features, context_labels,
                                                               features, context_labels, distance=args.proto_distance)
                    support_loss.backward()
                    optimizer.step()
                support_accs.append(support_acc.item())
                with torch.no_grad():
                    support_feature = context_features.unsqueeze(-1).unsqueeze(-1)
                    support_feature = F.conv2d(support_feature, vartheta[0]).flatten(1)
                    target_feature = target_features.unsqueeze(-1).unsqueeze(-1)
                    target_feature = F.conv2d(target_feature, vartheta[0]).flatten(1)
                target_loss, target_acc = loss_function(support_feature, context_labels,
                                                        target_feature, target_labels, distance=args.proto_distance)
                # print(j,'time, test:',support_loss.item(), support_acc.item())
                target_accs.append(target_acc.item())
            elif args.adaptor_model_name=='skip_based':
                adapt_model = skip_adaptor(base_model)
                adapt_model.to(device)
                adapt_model.reset()
                #print('we need train:', [k for k, v in adapt_model.named_parameters() if v.require_grad == True])
                optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, adapt_model.parameters()),
                                                 lr=lr)  # 有点疑问

                for test_i in range(args.epochs):
                    optimizer.zero_grad()
                    context_adapt_features = adapt_model.forward(context_features)
                    if_contains_nan(context_adapt_features)
                    # print(context_adapt_features)
                    support_loss, support_acc = loss_function(context_adapt_features, context_labels,
                                                              context_adapt_features, context_labels,
                                                              distance=args.proto_distance)
                    support_loss.backward()
                    optimizer.step()
                    # print(j,'time,',test_i,'epoch:', support_loss.item(), support_acc.item())
                support_accs.append(support_acc.item())
                with torch.no_grad():
                    support_feature = adapt_model.forward(context_features)
                    target_feature = adapt_model.forward(target_features)
                    target_loss, target_acc = loss_function(support_feature, context_labels,
                                                            target_feature, target_labels, distance=args.proto_distance)
                target_accs.append(target_acc.item())
            elif args.adaptor_model_name=='partial_based' or args.adaptor_model_name=='front_based' or args.adaptor_model_name=='full_based':
                adapt_model=base_model
                adapt_model.train()
                adapt_model.to(device)
                if args.adaptor_model_name=='partial_based':
                    layer_name='affine2'
                elif args.adaptor_model_name=='front_based':
                    layer_name='affine1'
                elif args.adaptor_model_name=='full_based':
                    layer_name='affine'
                for k,v in adapt_model.named_parameters():
                    if layer_name in k:
                        v.requires_grad=True
                    else:
                        v.requires_grad=False
                optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, adapt_model.parameters()),
                                                 lr=lr)
                """
                for test_i in range(args.epochs):
                    optimizer.zero_grad()
                    context_adapt_features = adapt_model.single_forward(context_features)
                    if_contains_nan(context_adapt_features)
                    # print(context_adapt_features)
                    support_loss, support_acc = loss_function(context_adapt_features, context_labels,
                                                              context_adapt_features, context_labels,
                                                              distance=args.proto_distance)
                    support_loss.backward()
                    optimizer.step()
                    # print(j,'time,',test_i,'epoch:', support_loss.item(), support_acc.item())
                support_accs.append(support_acc.item())
                with torch.no_grad():
                    support_feature = adapt_model.single_forward(context_features)
                    target_feature = adapt_model.single_forward(target_features)
                    target_loss, target_acc = loss_function(support_feature, context_labels,
                                                            target_feature, target_labels, distance=args.proto_distance)
                print(dataset,' test acc:',target_acc.item())
                target_accs.append(target_acc.item())
                """
                #epoch_params=[]
                target_acc_list=[]
                support_acc_list=[]
                for test_i in range(args.epochs):
                    optimizer.zero_grad()
                    context_adapt_features = adapt_model.single_forward(context_features)
                    #print('train_feature:',context_adapt_features.shape, context_adapt_features)
                    if_contains_nan(context_adapt_features)
                    # print(context_adapt_features)
                    support_loss, support_acc = loss_function(context_adapt_features, context_labels,
                                                              context_adapt_features, context_labels,
                                                              distance=args.proto_distance)
                    #support_loss, support_acc = cross_entropy_loss(context_adapt_features, context_labels)
                    if torch.isnan(support_loss).any() or torch.isnan(support_acc).any():
                        print('train feature has nan')
                        print('params:')
                        for k,v in adapt_model.named_parameters():
                            if layer_name in k:
                                print(k,v)
                        print('features:',context_adapt_features)
                        break
                    support_loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    support_feature = adapt_model.single_forward(context_features)
                    target_feature = adapt_model.single_forward(target_features)
                    target_loss, target_acc=loss_function(support_feature, context_labels,
                                                                  target_feature, target_labels,
                                                                  distance=args.proto_distance)
                    if j==args.test_size-1:
                        all_test_feature = torch.cat([support_feature, target_feature], dim=0)
                        test_feature_mean = torch.mean(all_test_feature, dim=0)
                        test_feature_norm = torch.norm(torch.var(all_test_feature, dim=0), p=2, dim=0)
                        dis = torch.norm(train_feature_mean - test_feature_mean, p=2, dim=0)
                        print(dataset, 'after train:dis:', dis.item(), ' train_norm:',
                            train_feature_norm, ' test_norm:', test_feature_norm)
                    if torch.isnan(target_loss).any() or torch.isnan(target_acc).any():
                        print('test feature has nan')
                        print('params:')
                        for k,v in adapt_model.named_parameters():
                            if layer_name in k:
                                print(k,v)
                        print('target features:',target_feature)
                        print('raw data:', target_features)
                        break
                support_accs.append(support_acc.item())
                target_accs.append(target_acc.item())
                """
                with torch.no_grad():
                    pred_list=[]
                    pred_accs=[]
                    for i in epoch_params:
                        adapt_model.load_state_dict(i, strict=False)
                        support_feature = adapt_model.single_forward(context_features)
                        target_feature = adapt_model.single_forward(target_features)
                        target_loss, target_acc, pred_dict = prototype_loss_with_preds(support_feature, context_labels,
                                                                                       target_feature, target_labels, distance=args.proto_distance)
                        pred_accs.append(target_acc.item())
                        pred_list.append(pred_dict['preds'])
                label_list=pred_dict['labels']
                pred_list=np.array(pred_list)
                vote_preds=[]
                for i in range(pred_list.shape[1]):
                    temp=pred_list[:,i]
                    pred=vote_for_preds(temp)
                    vote_preds.append(pred)
                vote_preds=np.array(vote_preds)
                preds=np.equal(vote_preds, label_list)
                target_acc1=np.mean(preds)
                target_acc2=max(pred_accs)
                target_acc=max([target_acc1, target_acc2])
                print(dataset, ' test acc:', target_acc1, target_acc2)
                target_accs.append(target_acc)
                """

            elif args.adaptor_model_name == 'affine_skip_based':
                if dataset in TRAIN_METADATASET_NAMES:
                    base_model.load_state_dict(seen_load_affine_model_params, strict=False)
                else:
                    base_model.load_state_dict(unseen_load_affine_model, strict=False)
                base_model.to(device)
                adapt_model = affine_skip_adaptor(base_model)
                adapt_model.to(device)
                adapt_model.reset()
                #print('we need train:', [k for k, v in adapt_model.named_parameters() if v.requires_grad == True])
                optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, adapt_model.parameters()),
                                                 lr=lr)  # 有点疑问

                for test_i in range(args.epochs):
                    optimizer.zero_grad()
                    context_adapt_features = adapt_model.forward(context_features)
                    if_contains_nan(context_adapt_features)
                    # print(context_adapt_features)
                    support_loss, support_acc = loss_function(context_adapt_features, context_labels,
                                                              context_adapt_features, context_labels,
                                                              distance=args.proto_distance)
                    support_loss.backward()
                    optimizer.step()
                    """
                    if dataset in TRAIN_METADATASET_NAMES:
                        t_index=TRAIN_METADATASET_NAMES.index(dataset)
                    else:
                        t_index=-1
                    if support_loss.item()<=train_valid_avg_loss_threshold[t_index]:
                        print(dataset,' train epochs:', test_i)
                        break
                    """
                    # print(j,'time,',test_i,'epoch:', support_loss.item(), support_acc.item())
                support_accs.append(support_acc.item())
                with torch.no_grad():
                    support_feature = adapt_model.forward(context_features)
                    target_feature = adapt_model.forward(target_features)
                    target_loss, target_acc = loss_function(support_feature, context_labels,
                                                            target_feature, target_labels, distance=args.proto_distance)
                print(dataset,' test acc:', target_acc.item())
                target_accs.append(target_acc.item())
        tag_name = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
        b = [0] * 10
        for i in range(len(target_accs)):
            temp = int(target_accs[i] * 10)
            if temp>=9:
                temp=9
            elif temp<=0:
                temp=0
            b[temp] += 1
        plt.bar(range(len(tag_name)), b, tick_label=tag_name)
        plt.title(dataset)
        plt.savefig(adaptor_result_output_path+'/'+dataset+'_'+str(args.epochs)+'.jpg')
        plt.close()
        test_support_acc[dataset]['mean'], test_support_acc[dataset]['var']=calculate_var(support_accs)
        test_target_acc[dataset]['mean'], test_target_acc[dataset]['var']=calculate_var(target_accs)
        print(dataset, ':')
        print('train_acc: ', test_support_acc[dataset]['mean'],'+-',test_support_acc[dataset]['var'])
        print('test_acc: ', test_target_acc[dataset]['mean'],'+-',test_target_acc[dataset]['var'])
session.close()
pretrain_info=args.pretrain_model_name
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
                        +universal_info+'_'+affine_info+'_'+adaptor_info+'_'+test_type+'_'+str(args.epochs)+'.json'
f=open(test_result_output_name, 'w',encoding='utf8')
f.write(final_info_json)
f.close()











