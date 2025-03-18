import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from six.moves import range
import gzip
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet the TensorFlow warnings
import tensorflow as tf
import pickle

def get_bscd_loader(dataset="EuroSAT", test_n_way=5, n_shot=5, image_size=224):
    iter_num = 400
    n_query = 10
    few_shot_params = dict(n_way=test_n_way , n_support=n_shot)

    if dataset == "EuroSAT":
        from data.extra_dataset.EuroSAT_few_shot import SetDataManager
    elif dataset == "ISIC":
        from data.extra_dataset.ISIC_few_shot import SetDataManager
    elif dataset == "CropDisease":
        from data.extra_dataset.CropDisease_few_shot import SetDataManager
    elif dataset == "ChestX":
        from data.extra_dataset.ChestX_few_shot import SetDataManager
    else:
        raise ValueError(f'Datast {dataset} is not supported.')

    datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
    novel_loader = datamgr.get_data_loader(aug=False)

    def _loader_wrap():
        for x, y in novel_loader:
            SupportTensor = x[:,:n_shot].contiguous().view(1, test_n_way*n_shot, *x.size()[2:])
            QryTensor = x[:, n_shot:].contiguous().view(1, test_n_way*n_query, *x.size()[2:])
            SupportLabel = torch.from_numpy( np.repeat(range( test_n_way ), n_shot) ).view(1, test_n_way*n_shot)
            QryLabel = torch.from_numpy( np.repeat(range( test_n_way ), n_query) ).view(1, test_n_way*n_query)

            yield SupportTensor, SupportLabel, QryTensor, QryLabel

    class _DummyGenerator:
        def manual_seed(self, seed):
            pass

    class _Loader(object):
        def __init__(self):
            self.iterable = _loader_wrap()
            # NOTE: the following are required by engine.py:_evaluate()
            self.dataset = self
            self.generator = _DummyGenerator()
        def __len__(self):
            return len(novel_loader)
        def __iter__(self):
            return self.iterable

    return _Loader()