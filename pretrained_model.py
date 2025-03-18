import os
import argparse
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import PROJECT_ROOT, TRAIN_METADATASET_NAMES, BATCHSIZES, device
from data.meta_dataset_reader import MetaDatasetBatchReader, MetaDatasetEpisodeReader
from model.base_network import ResNet, BaseNetwork
from model.model_utils import get_initial_optimizer, lr_scheduler, if_contains_nan
from model.model_loss import cross_entropy_loss,prototype_loss

#规定：所有单域模型只有最优的那个采用自己的数据集名字命名，其余的都带checkpoint或者epoch数作为后缀
#for example: omniglot dataset: best model: omniglot.pth.tar
#                               checkpoint: omniglot_checkpoint.pth.tar
#                               index_save: omniglot_10000.pth.tar
#checkpoint中保存：model参数, optimizer参数, 重要参数(epoch, best_val_acc, best_val_loss...)
#                schedular参数: 有epoch和args中的lr_schedular参数就能恢复
parser = argparse.ArgumentParser(description='pretrained model')
parser.add_argument('--dataset_index', type=int, default=0)

parser.add_argument('--if_inhert_url',type=bool, default=True)
parser.add_argument('--url_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/pretrained_model/url')
parser.add_argument('--if_restore_from_log', type=bool, default=True)
parser.add_argument('--model_ouput_base_path', type=str, default=PROJECT_ROOT+'/results/meta_train/pretrained_model/standard')

#model
parser.add_argument('--dropout', type=float, default=0.0)

#optimizer
parser.add_argument('--weight_decay',type=float,default=7e-4)
parser.add_argument('--learning_rate', type=float,default=0.01)
parser.add_argument('--optimizer_type',type=str,default='momentum')

#scheduler
parser.add_argument('--lr_scheduler', type=str, default='cosine')#cosine,step,exp

#train
parser.add_argument('--epochs', type=int, default=10000)

#valid
parser.add_argument('--valid_frequency', type=int, default=500)#其实也是checkpoint保存的频率
parser.add_argument('--valid_size', type=int, default=500)
parser.add_argument('--proto_distance', type=str, default='cos')#l2, cos
#test
parser.add_argument('--test_frequency',type=int, default=1000)#其实也是固定保存模型的频率
#record
parser.add_argument('--writer_frequency', type=int, default=200)
args = parser.parse_args()

for t_index, dataset in enumerate(TRAIN_METADATASET_NAMES):
#if args.dataset_index>=0:
 #   t_index=args.dataset_index
  #  dataset=TRAIN_METADATASET_NAMES[t_index]
    model_ouput_path=args.model_ouput_base_path+'/'+dataset
    writer=SummaryWriter(model_ouput_path)

    trainsets, valsets, testsets = [dataset], [dataset], [dataset]
    train_loader = MetaDatasetBatchReader('train', trainsets, valsets, testsets, batch_size=BATCHSIZES[dataset])
    val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)

    num_train_classes = train_loader.num_classes('train')
    model = ResNet(BaseNetwork, [2,2,2,2],num_classes=num_train_classes, dropout=args.dropout)

    #if inhert model parameters from url models
    if args.if_inhert_url==True:
        url_model_path=args.url_base_path+'/'+dataset+'/'+dataset+'.pth.tar'
        if os.path.isfile(url_model_path):
            url_ckpt=torch.load(url_model_path, map_location=device)['state_dict']
            model.load_state_dict(url_ckpt, strict=False)
            model.to(device)
        else:
            raise RuntimeError('there is no URL checkpoint in ', url_model_path)

    #set the optimizer(before the checkpoint load, since it may has the optimizer parameters)
    optimizer=get_initial_optimizer(model.get_parameters(), args)

    #if have checkpoint of this model, best_model or restorement
    if args.if_restore_from_log==True:
        checkpoint_model_path=model_ouput_path+'/'+dataset+'_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_model_path):
            checkpoint_ckpt = torch.load(checkpoint_model_path, map_location=device)
            model.load_state_dict(checkpoint_ckpt['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint_ckpt['optimizer'])
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
    else:
        restored_epoch = 0
        best_val_acc = 0.0
        best_val_loss = 99999

    #get the schedular
    scheduler=lr_scheduler(optimizer, restored_epoch, args)

    #record data
    epoch_train_loss, epoch_train_acc=[],[]
    epoch_val_loss, epoch_val_acc=[], []
    epoch_test_loss, epoch_test_acc=[], []

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=tf_config) as session:
        for epoch_i in tqdm(range(args.epochs), postfix='pretrain/train/'+dataset):
            if epoch_i<restored_epoch:
                continue

            optimizer.zero_grad()

            sample = train_loader.get_train_batch(session)
            #前面metadataset预处理的时候没有去除掉特别短的数据，有batch里只有1个的情况，会导致梯度中出现nan，这里要排除
            if sample['images'].size(0)<=1:
                print('batch data only contains 1 at epoch ', epoch_i)
                continue
            if_contains_nan(sample['images'])
            logits = model.forward(sample['images'])
            if_contains_nan(logits)
            if len(logits.size()) < 2:
                logits = logits.unsqueeze(0)
            train_loss, train_acc = cross_entropy_loss(logits, sample['labels'])
            epoch_train_loss.append(train_loss.item())
            epoch_train_acc.append(train_acc.item())

            train_loss.backward()
            optimizer.step()
            scheduler.step(epoch_i)

            if (epoch_i+1)%args.writer_frequency==0:
                writer.add_scalar(f"pretrain_train_loss/{dataset}", np.mean(epoch_train_loss), epoch_i+1)
                writer.add_scalar(f"pretrain_train_acc/{dataset}", np.mean(epoch_train_acc)*100, epoch_i+1)
                epoch_train_acc, epoch_train_loss = [], []
                writer.add_scalar(f"pretrain_learning_rate", optimizer.param_groups[0]['lr'], epoch_i+1)

            if (epoch_i+1)%args.valid_frequency==0:
                model.eval()
                val_losses, val_accs = [], []
                for j in tqdm(range(args.valid_size), postfix='pretrain/valid/'+dataset+'/'+str(epoch_i+1)+'_epoch'):
                    with torch.no_grad():
                        sample = val_loader.get_validation_task(session, dataset)
                        context_features = model.embed(sample['context_images'])
                        target_features = model.embed(sample['target_images'])
                        context_labels = sample['context_labels']
                        target_labels = sample['target_labels']
                        valid_loss, valid_acc = prototype_loss(context_features, context_labels, target_features, target_labels, args.proto_distance)
                    val_losses.append(valid_loss.item())
                    val_accs.append(valid_acc.item())
                valid_loss, valid_acc = np.mean(val_losses), np.mean(val_accs)*100
                epoch_val_loss.append(valid_loss)
                epoch_val_acc.append(valid_acc)
                writer.add_scalar(f"pretrain_valid_loss/{dataset}", valid_loss, epoch_i+1)
                writer.add_scalar(f"pretrain_valid_acc/{dataset}", valid_acc, epoch_i+1)

                if valid_acc>best_val_acc:
                    best_val_acc = valid_acc
                    best_val_loss = valid_loss
                    is_best = True
                    print('this is the best model so far!')
                else:
                    is_best = False

                checkpoint_model_path=model_ouput_path+'/'+dataset+'_checkpoint.pth.tar'
                information_dict={'state_dict':model.get_state_dict(),
                                  'optimizer':optimizer.state_dict(),
                                  'best_val_acc': best_val_acc,
                                  'best_val_loss':best_val_loss,
                                  'epoch':epoch_i+1,
                                  'lr_scheduler':args.lr_scheduler}

                torch.save(information_dict, checkpoint_model_path)
                if is_best:
                    best_model_path=model_ouput_path+'/'+dataset+'.pth.tar'
                    torch.save(information_dict, best_model_path)
                if (epoch_i+1)%args.test_frequency==0:
                    index_model_path=model_ouput_path+'/'+dataset+'_'+str(epoch_i+1)+'.pth.tar'
                    torch.save(information_dict, index_model_path)
                model.train()
    writer.close()