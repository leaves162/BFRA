import os
import argparse
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import PROJECT_ROOT, TRAIN_METADATASET_NAMES,ALL_METADATASET_NAMES, BATCHSIZES, device, KDANNEALING, LOSSWEIGHTS, KDFLOSSWEIGHTS, KDPLOSSWEIGHTS
from data.meta_dataset_reader import MetaDatasetBatchReader, MetaDatasetEpisodeReader
from model.base_network import ResNet, ResNet_MDL, ResNet_MDL_affine, BaseNetwork, BaseNetwork_affine
from model.model_utils import get_initial_optimizer, lr_scheduler, if_contains_nan, WeightAnnealing
from model.model_loss import cross_entropy_loss,prototype_loss, DistillKL,feature_difference_loss

parser = argparse.ArgumentParser(description='universal model')
parser.add_argument('--if_inhert_pretrain',type=bool, default=True)
parser.add_argument('--pretrain_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/pretrained_model/url')
parser.add_argument('--if_restore_from_log', type=bool, default=True)
parser.add_argument('--model_ouput_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/universal_part/feature_diff')
parser.add_argument('--model_affine_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/affine_part')
#model
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--if_affine_static', type=bool, default=False, help='确定是MDL还是MDL_affine模型框架')
parser.add_argument('--if_single_affine', type=bool, default=True, help='确定是多头affine还是单头')
#optimizer
parser.add_argument('--weight_decay',type=float,default=7e-4)
parser.add_argument('--learning_rate', type=float,default=0.001)
parser.add_argument('--optimizer_type',type=str,default='momentum')

#scheduler
parser.add_argument('--lr_scheduler', type=str, default='cosine')#cosine,step,exp

#train
parser.add_argument('--epochs', type=int, default=100000)#100000

#valid
parser.add_argument('--valid_frequency', type=int, default=1000)#1000, 其实也是checkpoint保存的频率
parser.add_argument('--valid_size', type=int, default=500)#500
parser.add_argument('--proto_distance', type=str, default='cos')#l2, cos
#test
parser.add_argument('--test_frequency',type=int, default=10000)#10000,#其实也是固定保存模型的频率
#record
parser.add_argument('--writer_frequency', type=int, default=200)#200
args = parser.parse_args()

trainsets, valsets, testsets = TRAIN_METADATASET_NAMES, TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES

train_loaders = []
num_train_classes = {}
kd_weight_annealing = {}
for t_index, trainset in enumerate(trainsets):
    train_loaders.append(MetaDatasetBatchReader('train', [trainset], valsets, testsets, batch_size=BATCHSIZES[trainset]))
    num_train_classes[trainset] = train_loaders[t_index].num_classes('train')
    kd_weight_annealing[trainset] = WeightAnnealing(T=int(args.valid_frequency*KDANNEALING[trainset]))
val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)

if args.if_affine_static:
    #是否把affine部分和universal部分一起训练，不一起训的话之后要固定universal单独训练affine
    model = ResNet_MDL(BaseNetwork, [2,2,2,2],num_classes=list(num_train_classes.values()), dropout=args.dropout)
    args.model_affine_base_path=args.model_affine_base_path+'/static_universal'
    args.model_ouput_base_path=args.model_ouput_base_path+'/none_affine_univsersal'
else:
    #确定一起训练还有affine单纯当一个层增强的，也有每个域单独一个，作为affine操作的
    model = ResNet_MDL_affine(BaseNetwork_affine, [2,2,2,2], num_classes=list(num_train_classes.values()),
                              dropout=args.dropout, if_static=args.if_single_affine)
    args.model_affine_base_path=args.model_affine_base_path+'/dynamic_universal'
    args.model_ouput_base_path=args.model_ouput_base_path+'/with_affine_universal'
    if args.if_single_affine:
        affine_name='single_head_affine'
    else:
        affine_name='multi_head_affine'
#if inhert model parameters from pretrain models
if args.if_inhert_pretrain == True:
    pretrain_model_path = args.pretrain_base_path + '/ilsvrc_2012/ilsvrc_2012.pth.tar'
    if os.path.isfile(pretrain_model_path):
        pretrain_ckpt = torch.load(pretrain_model_path, map_location=device)['state_dict']
        model.load_state_dict(pretrain_ckpt, strict=False)
        model.to(device)
    else:
        raise RuntimeError('there is no pretrain checkpoint in ', pretrain_model_path)

criterion_div = DistillKL(T=4)
optimizer=get_initial_optimizer(model.get_parameters(), args)

#if have checkpoint of this model, best_model or restorement
if args.if_restore_from_log==True:
    checkpoint_model_path=args.model_ouput_base_path+'/universal_part_checkpoint.pth.tar'
    if os.path.isfile(checkpoint_model_path):
        checkpoint_ckpt = torch.load(checkpoint_model_path, map_location=device)
        #print(checkpoint_ckpt.keys())
        model.load_state_dict(checkpoint_ckpt['state_dict'], strict=False)
        #optimizer.load_state_dict(checkpoint_ckpt['optimizer'])
        if args.lr_scheduler!=checkpoint_ckpt['lr_scheduler']:
            warnings.warn('the loaded optimizer from checkpoint is not match for the default optimizer!')
        restored_epoch=checkpoint_ckpt['epoch']
        best_val_loss=checkpoint_ckpt['best_val_loss']
        best_val_acc=checkpoint_ckpt['best_val_acc']
    else:
        restored_epoch=0
        best_val_acc=0.0
        best_val_loss=99999
        print('there is no model checkpoint yet, parameters are only initialized.')
    if args.if_affine_static==False:
        checkpoint_affine_path = args.model_affine_base_path + '/' + affine_name + '_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_affine_path):
            checkpoint_affine = torch.load(checkpoint_affine_path, map_location=device)
            model.load_state_dict(checkpoint_affine['affine'], strict=False)
        else:
            print('there is no affine checkpoint yet, parameters are only initialized.')
else:
    restored_epoch = 0
    best_val_acc = 0.0
    best_val_loss = 99999

#get the schedular
scheduler=lr_scheduler(optimizer, restored_epoch, args)

#load the single domain feature extractor(all of the seen domain pretrained model)
single_domian_feature_extractors = dict()
for t in trainsets:
    single_domian_pretrain_path=args.pretrain_base_path+'/'+t+'/'+t+'.pth.tar'
    single_extractor = ResNet(BaseNetwork, [2, 2, 2, 2], num_classes=num_train_classes[t], dropout=args.dropout)
    if os.path.isfile(single_domian_pretrain_path):
        url_ckpt = torch.load(single_domian_pretrain_path, map_location=device)['state_dict']
        single_extractor.load_state_dict(url_ckpt, strict=False)
        single_extractor.to(device)
        single_extractor.eval()
        single_domian_feature_extractors[t]=single_extractor

writer = SummaryWriter(args.model_ouput_base_path)

epoch_train_loss = {name: [] for name in trainsets}
epoch_train_acc = {name: [] for name in trainsets}
epoch_kd_f_loss = {name: [] for name in trainsets}
epoch_kd_p_loss = {name: [] for name in trainsets}
train_do_loss=[]
epoch_val_loss = {name: [] for name in valsets}
epoch_val_acc = {name: [] for name in valsets}

epoch_test_loss = {name:[] for name in testsets}
epoch_test_acc = {name:[] for name in testsets}

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = False
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
#tf_config.inter_op_parallelism_threads = 1  # 控制 TensorFlow 进程在执行跨操作并发时使用的线程数
#tf_config.intra_op_parallelism_threads = 1  # 控制 TensorFlow 进程在执行单个操作时使用的线程数

with tf.compat.v1.Session(config=tf_config) as session:
    for epoch_i in tqdm(range(args.epochs), postfix='universal/train/seen domains'):
        if epoch_i<restored_epoch:
            continue
        optimizer.zero_grad()
        samples = []
        images = dict()
        num_classes = []
        for t_index, (name, train_loader) in enumerate(zip(trainsets, train_loaders)):
            sample = train_loader.get_train_batch(session)
            samples.append(sample)
            images[name] = sample['images']
            num_classes.append(sample['images'].size(0))
        if_contains_nan(torch.cat(list(images.values())))
        #print(num_classes)
        mtl_logits, mtl_features = model.forward(torch.cat(list(images.values()),dim=0), num_classes, kd=True)
        flag=0
        for t_index, trainset in enumerate(trainsets):
            if mtl_features[t_index].shape[0]<=1:
                flag=1
                break
        if flag==1:
            print('batch data only contains 1 at epoch ', epoch_i)
            continue

        with torch.no_grad():
            stl_features = dict()
            stl_logits = dict()
            for name, single_extractor in single_domian_feature_extractors.items():
                stl_logits[name]=single_extractor.forward(images[name])
                stl_features[name]=single_extractor.embed(images[name])
        stl_features, stl_logits=list(stl_features.values()), list(stl_logits.values())

        train_losses = []
        kd_f_losses = 0
        kd_p_losses = 0
        for t_index, trainset in enumerate(trainsets):
            if_contains_nan(mtl_logits[t_index])
            if_contains_nan(mtl_features[t_index])
            if_contains_nan(stl_logits[t_index])
            if_contains_nan(stl_features[t_index])
            train_loss, train_acc = cross_entropy_loss(mtl_logits[t_index], samples[t_index]['labels'])
            train_losses.append(train_loss*LOSSWEIGHTS[trainset])

            epoch_train_loss[trainset].append(train_loss.item())
            epoch_train_acc[trainset].append(train_acc.item())

            f_teacher = torch.nn.functional.normalize(stl_features[t_index], p=2, dim=1, eps=1e-12)
            f_student = torch.nn.functional.normalize(mtl_features[t_index], p=2, dim=1, eps=1e-12)
            if_contains_nan(f_teacher)
            if_contains_nan(f_student)
            kd_f_loss = feature_difference_loss(f_teacher, f_student, samples[t_index]['labels'])
            kd_p_loss = criterion_div(mtl_logits[t_index], stl_logits[t_index])
            kd_weight = kd_weight_annealing[trainset](t=epoch_i, opt='linear')*KDFLOSSWEIGHTS[trainset]
            bam_weight = kd_weight_annealing[trainset](t=epoch_i, opt='linear')*KDPLOSSWEIGHTS[trainset]
            kd_f_losses += kd_f_loss*kd_weight
            kd_p_losses += kd_p_loss*bam_weight
            epoch_kd_f_loss[trainset].append(kd_f_loss.item())
            epoch_kd_p_loss[trainset].append(kd_p_loss.item())

        domain_label = []
        for i in range(len(num_classes)):
            temp_label = [i] * num_classes[i]
            domain_label = domain_label + temp_label
        domain_labels = torch.zeros(len(domain_label)).type(
            samples[0]['labels'].dtype).to(samples[0]['labels'].device)
        for i in range(len(domain_label)):
            domain_labels[i] = domain_label[i]
        DO_loss, DO_acc = prototype_loss(torch.cat(mtl_features, dim=0), domain_labels,
                                         torch.cat(mtl_features, dim=0), domain_labels, distance='cos')
        train_do_loss.append(DO_loss.item())

        train_loss = torch.stack(train_losses).sum()
        train_loss = train_loss+kd_p_losses+kd_f_losses+DO_loss.item()
        train_loss.backward()

        optimizer.step()
        scheduler.step(epoch_i)

        if (epoch_i+1)%args.valid_frequency==0:
            model.eval()
            epoch_val_acc, epoch_val_loss = [], []
            for valset in valsets:
                val_losses, val_accs = [], []
                for j in tqdm(range(args.valid_size), postfix='pretrain/valid/'+valset+'/'+str(epoch_i+1)+'_epoch'):
                    with torch.no_grad():
                        sample = val_loader.get_validation_task(session, valset)
                        if args.if_single_affine==True:
                            context_features = model.embed(sample['context_images'])
                            target_features = model.embed(sample['target_images'])
                        else:
                            context_num_samples=[0]*8
                            target_num_samples=[0]*8
                            val_index=TRAIN_METADATASET_NAMES.index(valset)
                            context_num_samples[val_index]+=sample['context_images'].size(0)
                            target_num_samples[val_index]+=sample['target_images'].size(0)
                            context_features = model.embed(sample['context_images'], context_num_samples)
                            target_features = model.embed(sample['target_images'], target_num_samples)
                        context_labels = sample['context_labels']
                        target_labels = sample['target_labels']
                        val_loss, val_acc = prototype_loss(context_features, context_labels, target_features, target_labels, args.proto_distance)
                    val_losses.append(val_loss.item())
                    val_accs.append(val_acc.item())
                val_acc, val_loss = np.mean(val_accs)*100, np.mean(val_losses)
                epoch_val_loss.append(val_loss)
                epoch_val_acc.append(val_acc)

                writer.add_scalar(f"universal_valid_loss/{valset}", val_loss, epoch_i+1)
                writer.add_scalar(f"universal_valid_acc/{valset}", val_acc, epoch_i+1)

            avg_val_loss, avg_val_acc = np.mean(epoch_val_loss), np.mean(epoch_val_acc)
            writer.add_scalar(f"universal_all_dataset_valid_loss_mean", avg_val_loss, epoch_i+1)
            writer.add_scalar(f"universal_all_dataset_valid_acc_mean", avg_val_acc, epoch_i+1)

            if avg_val_acc>best_val_acc:
                best_val_acc = avg_val_acc
                best_val_loss = avg_val_loss
                is_best = True
                print('this is the best model so far!')
            else:
                is_best = False

            checkpoint_model_path = args.model_ouput_base_path + '/universal_part_checkpoint.pth.tar'
            if args.if_affine_static:
                model_params = model.get_state_dict()
                affine_params = None#需要另外训练
            else:
                checkpoint_affine_path = args.model_affine_base_path + '/' + affine_name + '_checkpoint.pth.tar'
                model_params={}
                affine_params={}
                for k, v in model.get_state_dict().items():
                    if 'affine' in k:
                        affine_params[k]=v
                    else:
                        model_params[k]=v
                affine_dict = {'affine': affine_params}
                torch.save(affine_dict, checkpoint_affine_path)

            information_dict = {'state_dict': model_params,
                                'optimizer': optimizer.state_dict(),
                                'best_val_acc': best_val_acc,
                                'best_val_loss': best_val_loss,
                                'epoch': epoch_i + 1,
                                'lr_scheduler': args.lr_scheduler}
            torch.save(information_dict, checkpoint_model_path)

            # model和affine的命名方式都是保持一致的
            if is_best:
                best_model_path = args.model_ouput_base_path + '/universal_part.pth.tar'
                torch.save(information_dict, best_model_path)
                if args.if_affine_static==False:
                    best_affine_path = args.model_affine_base_path+'/'+affine_name+'.pth.tar'
                    torch.save(affine_dict, best_affine_path)
            if (epoch_i + 1) % args.test_frequency == 0:
                index_model_path = args.model_ouput_base_path + '/universal_part_' + str(epoch_i + 1) + '.pth.tar'
                torch.save(information_dict, index_model_path)
                if args.if_affine_static==False:
                    index_affine_path = args.model_affine_base_path + '/' + affine_name + '_' + str(
                        epoch_i + 1) + '.pth.tar'
                    torch.save(affine_dict, index_affine_path)
            model.train()

        if (epoch_i+1)%args.writer_frequency==0:
            for t in trainsets:
                writer.add_scalar(f"universal_train_loss/{t}", np.mean(epoch_train_loss[t]), epoch_i+1)
                writer.add_scalar(f"universal_train_acc/{t}", np.mean(epoch_train_acc[t]), epoch_i+1)
                writer.add_scalar(f"universal_kd_f_loss/{t}", np.mean(epoch_kd_f_loss[t]), epoch_i+1)
                writer.add_scalar(f"universal_kd_p_loss/{t}", np.mean(epoch_kd_p_loss[t]), epoch_i+1)
                epoch_train_acc[t], epoch_train_loss[t], epoch_kd_p_loss[t], epoch_kd_f_loss[t] = [], [], [], []
            writer.add_scalar(f"universal_learning_rate", optimizer.param_groups[0]['lr'], epoch_i+1)
            writer.add_scalar(f"train_do_loss", np.mean(train_do_loss), epoch_i+1)
            train_do_loss=[]
            checkpoint_model_path = args.model_ouput_base_path + '/universal_part_checkpoint.pth.tar'
            if args.if_affine_static:
                model_params = model.get_state_dict()
                affine_params = None  # 需要另外训练
            else:
                checkpoint_affine_path = args.model_affine_base_path + '/' + affine_name + '_checkpoint.pth.tar'
                model_params = {}
                affine_params = {}
                for k, v in model.get_state_dict().items():
                    if 'affine' in k:
                        affine_params[k] = v
                    else:
                        model_params[k] = v
                affine_dict = {'affine': affine_params}
                torch.save(affine_dict, checkpoint_affine_path)

            information_dict = {'state_dict': model_params,
                                'optimizer': optimizer.state_dict(),
                                'best_val_acc': best_val_acc,
                                'best_val_loss': best_val_loss,
                                'epoch': epoch_i + 1,
                                'lr_scheduler': args.lr_scheduler}
            torch.save(information_dict, checkpoint_model_path)

writer.close()

