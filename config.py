import argparse
import os
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT='your_path/code/BFRA'
#META_DATASET_ROOT = "/sda1/st_data/meta_dataset/raw"
META_RECORDS_ROOT = "your_path/meta_dataset/records"
META_SPLITS_ROOT = "your_path/meta_dataset/splits"
EXTRA_DATA_ROOT = 'your_path/extra_data_test'

ALL_METADATASET_NAMES = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco mnist cifar10 cifar100".split(' ')
TRAIN_METADATASET_NAMES = ALL_METADATASET_NAMES[:8]
VAL_METADATASET_NAMES = TRAIN_METADATASET_NAMES
TEST_METADATASET_NAMES = ALL_METADATASET_NAMES
#print(ALL_METADATASET_NAMES)
parser = argparse.ArgumentParser(description='Train prototypical networks')
parser.add_argument('--train_mode', type=str, default='MDL')
parser.add_argument('--pretrained_mode',type=str,default='MDL')
parser.add_argument('--if_pretrained', type=bool, default=True)
parser.add_argument('--if_augment', type=bool, default=False)
parser.add_argument('--pretrained_model_name', type=str, default='resnet18')
parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_models')
parser.add_argument('--result_models', type=str, default='./result_models')
parser.add_argument('--result_outputs', type=str, default='./result_outputs')

parser.add_argument('--train_datasets', type=list, default=TRAIN_METADATASET_NAMES)
parser.add_argument('--valid_datasets',type=list, default=VAL_METADATASET_NAMES)
parser.add_argument('--test_datasets', type=list, default=TEST_METADATASET_NAMES)

parser.add_argument('--num_workers',type=int, default=16)
parser.add_argument('--dropout',type=float, default=0.0)
parser.add_argument('--batch_size',type=int, default=32)
parser.add_argument('--epochs',type=int, default=500000)
parser.add_argument('--weight_decay',type=float,default=7e-4)
parser.add_argument('--learning_rate', type=float,default=0.001)
parser.add_argument('--lr_policy', type=str, default='cosine')#cosine,step,exp
parser.add_argument('--KL_f_weight',type=float,default=1.0)
parser.add_argument('--KL_p_weight',type=float,default=1.0)
parser.add_argument('--optimizer',type=str,default='momentum')#momentum,adam,ada

parser.add_argument('--valid_frequency',type=int, default=500)
parser.add_argument('--valid_size', type=int, default=300)
parser.add_argument('--writer_frequency',type=int,default=200)

parser.add_argument('--if_recover_from_log',type=bool,default=True) #resume
parser.add_argument('--test_frequency',type=int, default=10000)
parser.add_argument('--test_size', type=int, default=500)
parser.add_argument('--adaptor_mode',type=str, default='paramed')#metric,paramed,layer_paramed
parser.add_argument('--metric_mode',type=str, default='ncc',choices=['ncc', 'knn', 'lr', 'svm', 'scm'])#proto,var,proto_var
parser.add_argument('--metric_distance',type=str,default='cos',choices=['cos', 'l2'])#l2,cosine,ma
parser.add_argument('--metric_type', type=str, default='standard',choices=['standard', '1shot', '5shot'])#

parser.add_argument('--FT_learning_rate',type=float, default=0.01)

parser.add_argument('--adaptor.opt', type=str, default='linear', help="type of adaptor, linear or nonlinear")
args = vars(parser.parse_args())
# train args

batch_size=32
BATCHSIZES = {
                "ilsvrc_2012": batch_size*7,
                "omniglot": batch_size,
                "aircraft": batch_size,
                "cu_birds": batch_size,
                "dtd": batch_size,
                "quickdraw": batch_size,
                "fungi": batch_size,
                "vgg_flower": batch_size,
                "medmnist": batch_size
                }

LOSSWEIGHTS = {
                "ilsvrc_2012": 1,
                "omniglot": 1,
                "aircraft": 1,
                "cu_birds": 1,
                "dtd": 1,
                "quickdraw": 1,
                "fungi": 1,
                "vgg_flower": 1,
                "medmnist": 1
                }

# lambda^f in our paper
KDFLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1,
                    "medmnist": 1
                }
# lambda^p in our paper
KDPLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1,
                    "medmnist": 1
                }
# k in our paper
KDANNEALING = {
                    "ilsvrc_2012": 5,
                    "omniglot": 2,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 2,
                    "fungi": 2,
                    "vgg_flower": 1,
                    "medmnist": 1
                }

DATASET_MODELS_RESNET18 = {
    'ilsvrc_2012': 'imagenet-net',
    'omniglot': 'omniglot-net',
    'aircraft': 'aircraft-net',
    'cu_birds': 'birds-net',
    'dtd': 'textures-net',
    'quickdraw': 'quickdraw-net',
    'fungi': 'fungi-net',
    'vgg_flower': 'vgg_flower-net',
    'medmnist': 'medmnist-net'
}


DATASET_MODELS_RESNET18_PNF = {
    'omniglot': 'omniglot-film',
    'aircraft': 'aircraft-film',
    'cu_birds': 'birds-film',
    'dtd': 'textures-film',
    'quickdraw': 'quickdraw-film',
    'fungi': 'fungi-film',
    'vgg_flower': 'vgg_flower-film'
}

DATASET_MODELS_DICT = {'resnet18': DATASET_MODELS_RESNET18,
                       'resnet18_pnf': DATASET_MODELS_RESNET18_PNF}
