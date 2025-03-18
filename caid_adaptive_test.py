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
from model.adaptors import adaptor, tsa_adaptor, adapive_adaptor
from caid_test import paramed_test, layer_wise_paramed_test, calculate_test
from config import device, args, BATCHSIZES, LOSSWEIGHTS, KDFLOSSWEIGHTS, KDPLOSSWEIGHTS, KDANNEALING, DATASET_MODELS_DICT,TRAIN_METADATASET_NAMES, DATASET_MODELS_RESNET18
from model.network_adaptive import ResNet_MDL_adaptive

summary_out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
#writer = SummaryWriter(summary_out_path)
for t_index, trainset in enumerate(TRAIN_METADATASET_NAMES):
    trainsets, valsets, testsets = [trainset], [trainset], [trainset]
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['metric_type'])

    temp_mode=args['pretrained_mode']
    args['pretrained_mode']='SDL'
    model = get_initial_model(None, args)
    model = ResNet_MDL_adaptive(model)
    args['pretrained_mode']=temp_mode
    checkpointer = CheckPointer(args, model, optimizer=None)
    model_path='/multi-model-fusion-model-path/adaptive_'+trainset+'_best.pth.tar'
    ckpt=torch.load(model_path, map_location=device)
    #ckpt = torch.load(model_path, map_location=device)['state_dict']
    model.load_state_dict(ckpt, strict=False)
    model=model.to(device)
    model.eval()
    #for k, v in model.named_parameters():
     #   print(k)
    #break
    #model = adapive_adaptor(model)
    #model.reset()

    var_accs = dict()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        var_accs = paramed_test(model, testsets, trainsets, test_loader, session)

        out_path = args['result_outputs'] + '/' + args['train_mode'] + '_' + args['pretrained_model_name']
        out_path = os.path.join(out_path, '{}_{}_{}_{}_test_results_{}.npy'.format(args['adaptor_mode'],
                                                                                   args['metric_type'],
                                                                                   args['metric_mode'],
                                                                                   args['metric_distance'],
                                                                                   'adaptive_only'))
        test_accs = calculate_test(var_accs, testsets, out_path)

