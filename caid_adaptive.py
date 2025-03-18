import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from time import sleep
import tensorflow as tf
from tqdm import tqdm, trange
from tabulate import tabulate

from data.meta_dataset_reader import MetaDatasetBatchReader, MetaDatasetEpisodeReader
from model.losses import cross_entropy_loss, prototype_loss, distillation_loss, DistillKL, feature_gap_loss, scm_loss
from model.model_utils import UniformStepLR, CosineAnnealRestartLR, ExpDecayLR, WeightAnnealing
from model.get_model import get_initial_model, get_optimizer, multi_feature_extractor, CheckPointer
from model.pa import apply_selection, pa
from model.base_network import FT
from model.adaptors import adaptor, tsa_adaptor
from caid_test import paramed_test, layer_wise_paramed_test, calculate_test
from config import device, args, BATCHSIZES, LOSSWEIGHTS, KDFLOSSWEIGHTS, KDPLOSSWEIGHTS, KDANNEALING, DATASET_MODELS_DICT,TRAIN_METADATASET_NAMES, DATASET_MODELS_RESNET18
from model.network_adaptive import ResNet_MDL_adaptive
#trainsets, valsets, testsets = args['train_datasets'], args['valid_datasets'], args['test_datasets']
#train_loaders = []
#num_train_classes = {}
#kd_weight_annealing = {}
#for t_index, trainset in enumerate(trainsets):
 #   train_loaders.append(MetaDatasetBatchReader('train', [trainset], valsets, testsets, batch_size=BATCHSIZES[trainset]))
  #  num_train_classes[trainset] = train_loaders[t_index].num_classes('train')

summary_out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
writer = SummaryWriter(summary_out_path)
for t_index, trainset in enumerate(TRAIN_METADATASET_NAMES):
    trainsets, valsets, testsets = [trainset], [trainset], [trainset]
    train_loader = MetaDatasetBatchReader('train', trainsets, valsets, testsets,
                                          batch_size=args['batch_size'])
    val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)
    num_train_classes = train_loader.num_classes('train')
    temp_mode=args['pretrained_mode']
    args['pretrained_mode']='SDL'
    model = get_initial_model(num_train_classes, args)
    args['pretrained_mode']=temp_mode
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.load_model(mode='best', strict=False)
    model_path=checkpointer.temp_load_path+'/'+checkpointer.best_model_name
    ckpt=torch.load(model_path, map_location=device)
    #print(ckpt['state_dict'].keys())
    cls_params=model.state_dict()
    cls_params['cls_fn.weight']=ckpt['state_dict']['cls_fn.'+str(t_index)+'.weight']
    cls_params['cls_fn.bias']=ckpt['state_dict']['cls_fn.'+str(t_index)+'.bias']
    model.load_state_dict(cls_params)
    #print('pretrained: ',ckpt['state_dict']['cls_fn.'+str(t_index)+'.weight'])
    #print(model.state_dict()['cls_fn.weight'])
    #print('cls_fn.bias: ', model.named_parameters()['cls_fn.bias'])
    model.train()

    model=ResNet_MDL_adaptive(model)
    #model.reset()
    model=model.to(device)

    film_parameters = [v for k,v in model.named_parameters() if 'film' in k]
    params=[{'params':film_parameters}]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    epoch_loss = {name: [] for name in trainsets}
    epoch_acc = {name: [] for name in trainsets}

    epoch_val_loss = {name: [] for name in valsets}
    epoch_val_acc = {name: [] for name in valsets}

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False

    train_adaptive_epochs=50000
    best_val_acc=0
    with tf.compat.v1.Session(config=config) as session:
        for j in tqdm(range(train_adaptive_epochs), postfix=trainset):
            optimizer.zero_grad()
            model.zero_grad()

            sample = train_loader.get_train_batch(session)
            if sample['images'].size(0)<=1:
                print('batch data only 1 at step ', j)
                continue
            if torch.isnan(sample['images']).any():
                raise RuntimeError("image have nan values!!!")
            logits = model.forward(sample['images'])
            #logits = model.embed(sample['images'])
            if torch.isnan(logits).any():
                raise RuntimeError("logits have nan values!!!")
            #print(logits.shape)
            #print(sample['labels'])
            if len(logits.size()) < 2:
                logits = logits.unsqueeze(0)
            batch_loss, stats_dict, _ = cross_entropy_loss(logits, sample['labels'])

            batch_dataset = sample['dataset_name']
            epoch_loss[batch_dataset].append(stats_dict['loss'])
            epoch_acc[batch_dataset].append(stats_dict['acc'])
            batch_loss.backward()
            optimizer.step()

            if (j+1)%100==0:
                for dataset_name in trainsets:
                    writer.add_scalar(f"train_loss_adaptive/{dataset_name}", np.mean(epoch_loss[dataset_name]), j)
                    writer.add_scalar(f"train_accuracy_adaptive/{dataset_name}",np.mean(epoch_acc[dataset_name]), j)
                    epoch_loss[dataset_name], epoch_acc[dataset_name] = [], []
            if (j+1)%5000==0:
                print("\n==>Validing at {} epochs...".format(j + 1))
                model.eval()
                dataset_accs, dataset_losses = [], []
                for valset in valsets:
                    val_losses, val_accs = [], []
                    for k in range(args['valid_size']):
                        with torch.no_grad():
                            sample = val_loader.get_validation_task(session, valset)
                            context_features = model.embed(sample['context_images'])
                            target_features = model.embed(sample['target_images'])
                            context_labels = sample['context_labels']
                            target_labels = sample['target_labels']
                            _, state_dict, _ = prototype_loss(context_features, context_labels, target_features,
                                                              target_labels, args['metric_distance'])
                        val_losses.append(state_dict['loss'])
                        val_accs.append(state_dict['acc'])
                    dataset_acc, dataset_loss = np.mean(val_accs) * 100, np.mean(val_losses)
                    # print(dataset_acc, dataset_loss)
                    dataset_losses.append(dataset_loss)
                    dataset_accs.append(dataset_acc)
                    epoch_val_acc[valset].append(dataset_acc)
                    epoch_val_loss[valset].append(dataset_loss)

                    writer.add_scalar(f"val_loss_adaptive/{valset}", dataset_loss, j)
                    writer.add_scalar(f"val_accuracy_adaptive/{valset}", dataset_acc, j)
                avg_val_loss, avg_val_acc = np.mean(dataset_losses), np.mean(dataset_accs)
                if avg_val_acc>best_val_acc:
                    best_val_acc = avg_val_acc
                    best_val_loss = avg_val_loss
                    temp_model_path = checkpointer.temp_save_path + '/adaptive_' + trainset+'_best.pth.tar'
                    torch.save(model.state_dict(), temp_model_path)
                else:
                    temp_model_path = checkpointer.temp_save_path + '/adaptive_' + trainset + '_temp.pth.tar'
                    torch.save(model.state_dict(), temp_model_path)
    #torch.save(model.state_dict(), temp_model_path)
writer.close()