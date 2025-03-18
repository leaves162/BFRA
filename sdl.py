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
from model.losses import cross_entropy_loss, prototype_loss, distillation_loss, DistillKL, feature_gap_loss
from model.model_utils import UniformStepLR, CosineAnnealRestartLR, ExpDecayLR, WeightAnnealing
from model.get_model import get_initial_model, get_optimizer, multi_feature_extractor, CheckPointer
from model.pa import apply_selection, pa
from model.base_network import FT
from model.adaptors import adaptor, tsa_adaptor
from caid_test import paramed_test, layer_wise_paramed_test, calculate_test
from config import device, args, BATCHSIZES, LOSSWEIGHTS, KDFLOSSWEIGHTS, KDPLOSSWEIGHTS, KDANNEALING, DATASET_MODELS_DICT,TRAIN_METADATASET_NAMES

temp_model_mode = args['train_mode']
args['train_mode'] = 'SDL'
args['pretrained_mode'] = 'SDL'
summary_out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
writer = SummaryWriter(summary_out_path)
for t_index, trainset in enumerate(TRAIN_METADATASET_NAMES):
    trainsets, valsets, testsets = [trainset], [trainset], [trainset]
    train_loader = MetaDatasetBatchReader('train', trainsets, valsets, testsets,
                                          batch_size=args['train.batch_size'])
    val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)
    num_train_classes = train_loader.num_classes('train')
    model = get_initial_model(num_train_classes, args, trainset)
    optimizer = get_optimizer(model, args, params=model.get_parameters())
    checkpointer = CheckPointer(args, model, optimizer=optimizer)

    if os.path.isfile(checkpointer.temp_load_path+'/'+checkpointer.temp_model_name) and args['if_recover_from_log']:
        start_epoch, best_val_loss, best_val_acc = checkpointer.load_model()
    else:
        print('Load checkpoint: there is no checkpoint for ',checkpointer.temp_load_path+'/'+checkpointer.temp_model_name)
        best_val_loss = 99999
        start_epoch = 0
        best_val_acc = 0

    if args['lr_policy'] == "step":
        lr_manager = UniformStepLR(optimizer, args, start_epoch)
    elif args['lr_policy'] == "exp":
        lr_manager = ExpDecayLR(optimizer, args, start_epoch)
    elif args['lr_policy'] == "cosine":
        lr_manager = CosineAnnealRestartLR(optimizer, args, start_epoch)

    max_epoch = args['epochs']
    epoch_loss = {name: [] for name in trainsets}
    epoch_acc = {name: [] for name in trainsets}

    epoch_val_loss = {name: [] for name in valsets}
    epoch_val_acc = {name: [] for name in valsets}

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=tf_config) as session:
        for i in tqdm(range(max_epoch)):
            #print(i)
            if i<start_epoch:
                continue

            optimizer.zero_grad()

            sample = train_loader.get_train_batch(session)
            if sample['images'].size(0)<=1:
                print('batch data only 1 at step ', i)
                continue
            if torch.isnan(sample['images']).any():
                raise RuntimeError("image have nan values!!!")
            logits = model.forward(sample['images'])
            if torch.isnan(logits).any():
                raise RuntimeError("logits have nan values!!!")
            if len(logits.size()) < 2:
                logits = logits.unsqueeze(0)
            batch_loss, stats_dict, _ = cross_entropy_loss(logits, sample['labels'])
            batch_dataset = sample['dataset_name']
            epoch_loss[batch_dataset].append(stats_dict['loss'])
            epoch_acc[batch_dataset].append(stats_dict['acc'])

            batch_loss.backward()
            optimizer.step()
            lr_manager.step(i)

            if (i+1)%args['writer_frequency']==0:
                for dataset_name in trainsets:
                    writer.add_scalar(f"train_loss/{dataset_name}", np.mean(epoch_loss[dataset_name]), i)
                    writer.add_scalar(f"train_accuracy/{dataset_name}",np.mean(epoch_acc[dataset_name]), i)
                    epoch_loss[dataset_name], epoch_acc[dataset_name] = [], []
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i)

            if (i+1)%args['valid_frequency']==0:
                print("\n==>Validing at {} epochs...".format(i+1))
                model.eval()
                dataset_accs, dataset_losses = [], []
                for valset in valsets:
                    val_losses, val_accs = [], []
                    for j in range(args['valid_size']):
                        with torch.no_grad():
                            sample = val_loader.get_validation_task(session, valset)
                            context_features = model.embed(sample['context_images'])
                            target_features = model.embed(sample['target_images'])
                            context_labels = sample['context_labels']
                            target_labels = sample['target_labels']
                            _, state_dict, _ = prototype_loss(context_features, context_labels, target_features, target_labels, args['metric_distance'])
                        val_losses.append(state_dict['loss'])
                        val_accs.append(state_dict['acc'])
                    dataset_acc, dataset_loss = np.mean(val_accs)*100, np.mean(val_losses)
                    #print(dataset_acc, dataset_loss)
                    dataset_losses.append(dataset_loss)
                    dataset_accs.append(dataset_acc)
                    epoch_val_acc[valset].append(dataset_acc)
                    epoch_val_loss[valset].append(dataset_loss)

                    writer.add_scalar(f"val_loss/{valset}", dataset_loss, i)
                    writer.add_scalar(f"val_accuracy/{valset}", dataset_acc, i)
                    #print(f"{valset}: val_acc {dataset_acc:.2f}%, val_loss {dataset_loss:.3f}")

                avg_val_loss, avg_val_acc = np.mean(dataset_losses), np.mean(dataset_accs)
                writer.add_scalar(f"mean_val_loss/avg_val_loss", avg_val_loss, i)
                writer.add_scalar(f"mean_val_accuracy/avg_val_acc", avg_val_acc, i)

                if avg_val_acc>best_val_acc:
                    best_val_acc = avg_val_acc
                    best_val_loss = avg_val_loss
                    is_best = True
                    print('this is the best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_loss': epoch_loss,
                              'epoch_acc': epoch_acc,
                              'epoch_val_loss': epoch_val_loss,
                              'epoch_val_acc': epoch_val_acc}
                checkpointer.save_model(is_best=is_best, epoch=i, best_val_acc=best_val_acc, best_val_loss=best_val_loss,
                                        new_model_state=model.get_state_dict(), new_optimizer_state=optimizer.state_dict(), extra_info=extra_dict)
                if (i+1)%args['test_frequency']==0:
                        checkpointer.save_model_index(epoch=i+1, best_val_acc=best_val_acc, best_val_loss=best_val_loss,index=i+1,
                                        new_model_state=model.get_state_dict(), new_optimizer_state=optimizer.state_dict(), extra_info=extra_dict)
                model.train()
            if (i+1)%args['test_frequency']==0:
                print("\n==>Testing at {} epochs...".format(i + 1))
                test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets,
                                                       test_type=args['metric_type'])

                model.eval()
                if args['adaptor_mode'] == 'layer_paramed':
                    model = tsa_adaptor(model)
                    model.reset()
                if args['adaptor_mode']=='paramed':
                    var_accs = paramed_test(model, testsets, trainsets, test_loader, session)
                if args['adaptor_mode']=='layer_paramed':
                    var_accs = layer_wise_paramed_test(model, testsets, trainsets, test_loader, session)

                out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
                out_path = os.path.join(out_path, '{}_{}_{}_{}_test_results_{}.npy'.format(args['adaptor_mode'],
                                                                                           args['metric_type'],
                                                                                           args['metric_mode'],
                                                                                           args['metric_distance'],
                                                                                           i + 1))
                test_accs = calculate_test(var_accs, testsets, out_path)
                for d in range(len(testsets)):
                    writer.add_scalar(f"test_accuracy/{testsets[d]}", test_accs[d], i)
                model.train()
    if start_epoch < max_epoch:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, best_avg_val_acc: {best_val_acc:.2f}%""")
    else:
        print(f"""No training happened. Loaded checkpoint at {start_epoch}, while max_iter was {max_epoch}""")
writer.close()
args['train_mode'] = temp_model_mode
args['pretrained_mode'] = temp_model_mode
