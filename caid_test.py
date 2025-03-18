import os
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir

from model.losses import prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss
from model.get_model import get_initial_model, CheckPointer
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader)
from config import args, ALL_METADATASET_NAMES, TRAIN_METADATASET_NAMES
from model.adaptors import adaptor, layer_wise_adaptor, feature_transform, tsa_adaptor, layer_wise_adaptor_adaptive

def paramed_test(model, testsets, trainsets, test_loader, session):
    var_accs = dict()
    for datasett in testsets:
        if datasett in trainsets:
            lr = 0.1
        else:
            lr = 1
        var_accs[datasett] = []
        for j in tqdm(range(args['test_size']),postfix=datasett):
            with torch.no_grad():
                sample = test_loader.get_test_task(session, datasett)
                context_features = model.embed(sample['context_images'])
                target_features = model.embed(sample['target_images'])
                context_labels = sample['context_labels']
                target_labels = sample['target_labels']
            # optimize selection parameters and perform feature selection
            selection_params = adaptor(context_features, context_labels, max_iter=50, lr=lr,
                                  distance=args['metric_distance'])
            selected_context = feature_transform(context_features, selection_params)
            selected_target = feature_transform(target_features, selection_params)
            if args['metric_mode']=='ncc':
                _, stats_dict, _ = prototype_loss(
                    selected_context, context_labels,
                    selected_target, target_labels, distance=args['metric_distance'])
            if args['metric_mode']=='scm':
                _, stats_dict, _ = scm_loss(
                    selected_context, context_labels,
                    selected_target, target_labels, normalize=True)
            var_accs[datasett].append(stats_dict['acc'])
    return var_accs

def layer_wise_paramed_test(model, testsets, trainsets, test_loader, session):
    var_accs = dict()
    for dataset in testsets:
        if dataset in trainsets:
            lr = 0.05
            lr_beta = 0.1
        else:
            lr = 0.5
            lr_beta = 1
        var_accs[dataset] = []
        for j in tqdm(range(args['test_size']),postfix=dataset):
            # initialize task-specific adapters and pre-classifier alignment for each task
            model.reset()
            # loading a task containing a support set and a query set
            sample = test_loader.get_test_task(session, dataset)
            context_images = sample['context_images']
            target_images = sample['target_images']
            context_labels = sample['context_labels']
            target_labels = sample['target_labels']
            # optimize task-specific adapters and/or pre-classifier alignment
            ###############################################################

            layer_wise_adaptor(context_images, context_labels, model, max_iter=40, lr=lr, lr_beta=lr_beta,
                distance=args['metric_distance'])
            with torch.no_grad():
                context_features = model.embed(sample['context_images'])
                target_features = model.embed(sample['target_images'])
                #####################################################
                #context_features = model.se_beta(context_features)
                #target_features = model.se_beta(target_features)
                context_features = model.beta(context_features)
                target_features = model.beta(target_features)
            if args['metric_mode'] == 'ncc':
                _, stats_dict, _ = prototype_loss(
                    context_features, context_labels,
                    target_features, target_labels, distance=args['metric_distance'])
            if args['metric_mode'] == 'scm':
                _, stats_dict, _ = scm_loss(
                    context_features, context_labels,
                    target_features, target_labels, normalize=True)
            var_accs[dataset].append(stats_dict['acc'])
    return var_accs

def calculate_test(var_accs, testsets, out_path):
    rows = []
    accs = []
    for dataset_name in testsets:
        row = [dataset_name]
        acc = np.array(var_accs[dataset_name]) * 100
        mean_acc = acc.mean()
        accs.append(mean_acc)
        conf = (1.96 * acc.std()) / np.sqrt(len(acc))
        row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        #writer.add_scalar(f"test_accuracy/{dataset_name}", mean_acc, i)
        rows.append(row)
    #out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
    # out_path = check_dir(out_path, True)
    #out_path = os.path.join(out_path, '{}_{}_{}_{}_test_results_{}.npy'.format(args['adaptor_mode'],
                                                                               #args['metric_type'],
                                                                               #args['metric_mode'],
                                                                               #args['metric_distance'],
                                                                               #i + 1))
    np.save(out_path, {'rows': rows})
    table = tabulate(rows, headers=['model \\ data'] + ['test_acc'], floatfmt=".2f")
    print(table)
    print('\n')
    return accs

def test():
    # Setting up datasets
    trainsets, valsets, testsets = args['train_datasets'], args['valid_datasets'], args['test_datasets']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['metric_type'])
    model = get_initial_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.load_model(mode='best',strict=False)
    model.eval()
    if args['adaptor_mode']=='layer_paramed':
        model =tsa_adaptor(model)
        model.reset()
    var_accs = dict()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        if args['adaptor_mode'] == 'paramed':
            var_accs = paramed_test(model, testsets, trainsets, test_loader, session)
        if args['adaptor_mode'] == 'layer_paramed':
            var_accs = layer_wise_paramed_test(model, testsets, trainsets, test_loader, session)

        out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
        out_path = os.path.join(out_path, '{}_{}_{}_{}_test_results_{}.npy'.format(args['adaptor_mode'],
                                                                                   args['metric_type'],
                                                                                   args['metric_mode'],
                                                                                   args['metric_distance'],
                                                                                   'only'))
        test_accs = calculate_test(var_accs, testsets, out_path)
    return

def test_with_specific_model(model_index):
    # Setting up datasets
    print('testing with specific model index: ', model_index,' ...')
    trainsets, valsets, testsets = args['train_datasets'], args['valid_datasets'], args['test_datasets']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['metric_type'])
    model = get_initial_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.load_model_index(mode='last',index=model_index, strict=False)
    model.eval()
    if args['adaptor_mode']=='layer_paramed':
        model =tsa_adaptor(model)
        model.reset()
    var_accs = dict()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        if args['adaptor_mode'] == 'paramed':
            var_accs = paramed_test(model, testsets, trainsets, test_loader, session)
        if args['adaptor_mode'] == 'layer_paramed':
            var_accs = layer_wise_paramed_test(model, testsets, trainsets, test_loader, session)

        out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
        out_path = os.path.join(out_path, '{}_{}_{}_{}_test_results_{}.npy'.format(args['adaptor_mode'],
                                                                                   args['metric_type'],
                                                                                   args['metric_mode'],
                                                                                   args['metric_distance'],
                                                                                   str(model_index)))
        test_accs = calculate_test(var_accs, testsets, out_path)
    return

def test_all_index_model():
    all_model_files = os.listdir('/sda1/st_data/code/CAID_2/result_models/MDL_resnet18')
    model_files = []
    for i in all_model_files:
        if '0000' in i:
            model_files.append(int(i[11:-8]))
    model_files.sort()
    print('Now model index has:', model_files)
    for i in model_files:
        test_with_specific_model(i)
if __name__ == '__main__':
    #test()
    test_all_index_model()




