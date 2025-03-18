import argparse
import os
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT='/sda1/st_data/code/TP_FDAS'
META_DATASET_ROOT = "/sda1/st_data/meta_dataset/raw"
META_RECORDS_ROOT = "/sda1/st_data/meta_dataset/records"



parser = argparse.ArgumentParser(description='Train prototypical networks')
ALL_METADATASET_NAMES = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco mnist cifar10 cifar100".split(' ')
TRAIN_METADATASET_NAMES = ALL_METADATASET_NAMES[:8]
VAL_METADATASET_NAMES=TRAIN_METADATASET_NAMES
TEST_METADATASET_NAMES = ALL_METADATASET_NAMES

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

parser.add_argument('--num_workers',type=int, default=32)
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

# train args

parser.add_argument('--train.optimizer', type=str, default='momentum', metavar='OPTIM',
                    help='optimization method (default: momentum)')


parser.add_argument('--train.lr_decay_step_gamma', type=int, default=1e-1, metavar='DECAY_GAMMA',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.lr_decay_step_freq', type=int, default=10000, metavar='DECAY_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_final_lr', type=float, default=8e-5, metavar='FINAL_LR',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_start_iter', type=int, default=30000, metavar='START_ITER',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.cosine_anneal_freq', type=int, default=4000, metavar='ANNEAL_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.nesterov_momentum', action='store_true', help="If to augment query images in order to avearge the embeddings")


# evaluation during training




# creating a database of features
parser.add_argument('--dump.name', type=str, default='', metavar='DUMP_NAME',
                    help='Name for dumped dataset of features')
parser.add_argument('--dump.mode', type=str, default='test', metavar='DUMP_MODE',
                    help='What split of the original dataset to dump')
parser.add_argument('--dump.size', type=int, default=600, metavar='DUMP_SIZE',
                    help='Howe many episodes to dump')


# test args
parser.add_argument('--test.size', type=int, default=600, metavar='TEST_SIZE',
                    help='The number of test episodes sampled')
parser.add_argument('--test.mode', type=str, choices=['mdl', 'sdl'], default='mdl', metavar='TEST_MODE',
                    help="Test mode: multi-domain learning (mdl) or single-domain learning (sdl) settings")
parser.add_argument('--test.type', type=str, choices=['standard', '1shot', '5shot'], default='standard', metavar='LOSS_FN',
                    help="meta-test type, standard varying number of ways and shots as in Meta-Dataset, 1shot for five-way-one-shot and 5shot for varying-way-five-shot evaluation.")
parser.add_argument('--test.distance', type=str, choices=['cos', 'l2'], default='cos', metavar='DISTANCE_FN',
                    help="feature similarity function")
parser.add_argument('--test.loss-opt', type=str, choices=['ncc', 'knn', 'lr', 'svm', 'scm'], default='ncc', metavar='LOSS_FN',
                    help="Loss function for meta-testing, knn or prototype loss (ncc), Support Vector Machine (svm), Logistic Regression (lr) or Mahalanobis Distance (scm)")
parser.add_argument('--test.feature-norm', type=str, choices=['l2', 'none'], default='none', metavar='LOSS_FN',
                    help="normalization options")

# task-specific adapters
parser.add_argument('--test.tsa-ad-type', type=str, choices=['residual', 'serial', 'none'], default='none', metavar='TSA_AD_TYPE',
                    help="adapter type")
parser.add_argument('--test.tsa-ad-form', type=str, choices=['matrix', 'vector', 'none'], default='matrix', metavar='TSA_AD_FORM',
                    help="adapter form")
parser.add_argument('--test.tsa-opt', type=str, choices=['alpha', 'beta', 'alpha+beta'], default='alpha+beta', metavar='TSA_OPT',
                    help="task adaptation option")
parser.add_argument('--test.tsa-init', type=str, choices=['random', 'eye'], default='eye', metavar='TSA_INIT',
                    help="initialization for adapter")

# path args
parser.add_argument('--model.dir', default='your_path/BFRA/result_models', type=str, metavar='PATH',
                    help='path of single domain learning models')
parser.add_argument('--out.dir', default='your_path/BFRA/result_models', type=str, metavar='PATH',
                    help='directory to output the result and checkpoints')
parser.add_argument('--source', default='your_path/BFRA/result_models/SDL', type=str, metavar='PATH',
                    help='path of pretrained model')

# log args
args = vars(parser.parse_args())

BATCHSIZES = {
                "ilsvrc_2012": 448,
                "omniglot": 64,
                "aircraft": 64,
                "cu_birds": 64,
                "dtd": 64,
                "quickdraw": 64,
                "fungi": 64,
                "vgg_flower": 64
                }

LOSSWEIGHTS = {
                "ilsvrc_2012": 1,
                "omniglot": 1,
                "aircraft": 1,
                "cu_birds": 1,
                "dtd": 1,
                "quickdraw": 1,
                "fungi": 1,
                "vgg_flower": 1
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
                    "vgg_flower": 1
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
                    "vgg_flower": 1
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
                    "vgg_flower": 1
                }

DATASET_MODELS_RESNET18 = {
    'ilsvrc_2012': 'imagenet-net',
    'omniglot': 'omniglot-net',
    'aircraft': 'aircraft-net',
    'cu_birds': 'birds-net',
    'dtd': 'textures-net',
    'quickdraw': 'quickdraw-net',
    'fungi': 'fungi-net',
    'vgg_flower': 'vgg_flower-net'
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
