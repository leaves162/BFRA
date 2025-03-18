"""
1.对初始化model加载的函数，optimizer加载的函数
2.日志重载类checkpointer
3.多个单域网络加载并dict或list形式输出函数
"""

import os
import torch
import shutil
from functools import partial
from config import device, DATASET_MODELS_RESNET18

class CheckPointer(object):
    def __init__(self, args, model=None, optimizer=None):
        #self.mode=mode#restore, load
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.initial_load_path=args['pretrained_model_path']+'/'+args['pretrained_mode']+'_'+args['pretrained_model_name']
        self.temp_load_path=args['result_models']+'/'+args['train_mode']+'_'+args['pretrained_model_name']
        self.temp_save_path=self.temp_load_path
        self.temp_model_name='checkpoint.pth.tar'
        self.best_model_name='model_best.pth.tar'

    def save_model(self, is_best, epoch, best_val_acc, best_val_loss,
                   new_model_state=None,new_optimizer_state=None, extra_info=None):
        if self.model is not None:
            if new_model_state is not None:
                model_state_dict=new_model_state
            else:
                model_state_dict=self.model.state_dict()
        else:
            if new_model_state is not None:
                model_state_dict=new_model_state
            else:
                model_state_dict=None

        if self.optimizer is not None:
            if new_optimizer_state is not None:
                optimizer_state_dict=new_optimizer_state
            else:
                optimizer_state_dict=self.optimizer.state_dict()
        else:
            if new_optimizer_state is not None:
                optimizer_state_dict=new_optimizer_state
            else:
                optimizer_state_dict=None

        state_info={'epoch':epoch,
                    'args':self.args,
                    'state_dict':model_state_dict,
                    'optimizer':optimizer_state_dict,
                    'best_val_loss':best_val_loss,
                    'best_val_acc':best_val_acc}
        if extra_info is not None:
            state_info.update(extra_info)
        temp_model_path=self.temp_save_path+'/'+self.temp_model_name
        best_model_path=self.temp_save_path+'/'+self.best_model_name
        torch.save(state_info, temp_model_path)
        print('Save checkpoint: temp model saves at ', temp_model_path)
        if is_best is True:
            shutil.copyfile(temp_model_path, best_model_path)
            print('Save checkpoint: best model saves at ',best_model_path)

    def load_model(self,mode='last',model=True,optimizer=True, if_initial=False, dataset_name=None,strict=True):
        if if_initial==False and dataset_name==None:
            if mode=='last':
                model_path=self.temp_load_path+'/'+self.temp_model_name
            else:
                model_path=self.temp_load_path+'/'+self.best_model_name
        else:
            model_path=self.initial_load_path+'/'+DATASET_MODELS_RESNET18[dataset_name]+'/'+self.best_model_name
        if os.path.isfile(model_path):
            print("Load checkpoint: {} checkpoint from '{}'".format(mode, model_path))
            ckpt=torch.load(model_path, map_location=device)
            #print('cls: ',ckpt['state_dict']['cls_fn.6.bias'])
            #print([k for k, v in self.model.named_parameters()])
            if self.model is not None and model:
                self.model.load_state_dict(ckpt['state_dict'],strict=strict)
            else:
                print('initial model is None which dismatch loaded state_dict.')
            if self.optimizer is not None and optimizer:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            else:
                print('Warning: initial optimizer is None or dismatch loaded optimizer.')
        else:
            print('Load checkpoint: there is no checkpoint for ',model_path)
        return ckpt.get('epoch', None), ckpt.get('best_val_loss', None), ckpt.get('best_val_acc', None)

    def save_model_index(self, epoch, best_val_acc, best_val_loss,index,
                   new_model_state=None,new_optimizer_state=None, extra_info=None):
        if self.model is not None:
            if new_model_state is not None:
                model_state_dict=new_model_state
            else:
                model_state_dict=self.model.state_dict()
        else:
            if new_model_state is not None:
                model_state_dict=new_model_state
            else:
                model_state_dict=None

        if self.optimizer is not None:
            if new_optimizer_state is not None:
                optimizer_state_dict=new_optimizer_state
            else:
                optimizer_state_dict=self.optimizer.state_dict()
        else:
            if new_optimizer_state is not None:
                optimizer_state_dict=new_optimizer_state
            else:
                optimizer_state_dict=None

        state_info={'epoch':epoch,
                    'args':self.args,
                    'state_dict':model_state_dict,
                    'optimizer':optimizer_state_dict,
                    'best_val_loss':best_val_loss,
                    'best_val_acc':best_val_acc}
        if extra_info is not None:
            state_info.update(extra_info)
        temp_model_path=self.temp_save_path+'/checkpoint_'+str(index)+'.pth.tar'
        #best_model_path=self.temp_save_path+'/'+self.best_model_name
        torch.save(state_info, temp_model_path)
        print('Save checkpoint: specific epoch ',index,' model saves at ', temp_model_path)

    def load_model_index(self,mode='last',index=10000, model=True,optimizer=True, if_initial=False, dataset_name=None,strict=True):
        if if_initial==False and dataset_name==None:
            if mode=='last':
                model_path=self.temp_load_path+'/checkpoint_'+str(index)+'.pth.tar'
            else:
                model_path=self.temp_load_path+'/'+self.best_model_name
        else:
            model_path=self.initial_load_path+'/'+DATASET_MODELS_RESNET18[dataset_name]+'/'+self.best_model_name
        if os.path.isfile(model_path):
            print("Load checkpoint: specific checkpoint from '{}'".format(model_path))
            ckpt=torch.load(model_path, map_location=device)
            if self.model is not None and model:
                self.model.load_state_dict(ckpt['state_dict'],strict=strict)
            else:
                print('initial model is None which dismatch loaded state_dict.')
            if self.optimizer is not None and optimizer:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            else:
                print('Warning: initial optimizer is None or dismatch loaded optimizer.')
        else:
            print('Load checkpoint: there is no checkpoint for ',model_path)
        return ckpt.get('epoch', None), ckpt.get('best_val_loss', None), ckpt.get('best_val_acc', None)

def get_initial_model(num_classes, args,dataset_name='ilsvrc_2012'):
    train_classifier = 'linear'
    model_name = args['pretrained_model_name']
    model_mode = args['pretrained_mode']
    if_aug = args['if_augment']
    dropout = args.get('dropout', 0)
    #print(num_classes)
    if args['if_pretrained'] and dataset_name is not None:
        base_network_name = DATASET_MODELS_RESNET18[dataset_name]
        base_network_path = args['pretrained_model_path']+'/'+model_mode+'_'+model_name+'/'+base_network_name+'/'+'model_best.pth.tar'
    if if_aug==False:
        if model_name=='resnet18' and model_mode=='SDL':
            from model.base_network import resnet18
            if args['if_pretrained'] and dataset_name is not None:
                model_fn = partial(resnet18, dropout=dropout,pretrained=args['if_pretrained'], pretrained_model_path=base_network_path)
            else:
                model_fn = partial(resnet18, dropout=dropout)
        elif model_name=='resnet18' and model_mode=='MDL':
            from model.base_network import resnet18_mdl
            if args['if_pretrained'] and dataset_name is not None:
                model_fn = partial(resnet18_mdl, dropout=dropout,pretrained=args['if_pretrained'],pretrained_model_path=base_network_path)
            else:
                model_fn = partial(resnet18_mdl, dropout=dropout)
        elif model_name=='resnet50' and model_mode=='SDL':
            from model.base_network import resnet50
            if args['if_pretrained'] and dataset_name is not None:
                model_fn = partial(resnet50, dropout=dropout,pretrained=args['if_pretrained'],pretrained_model_path=base_network_path)
            else:
                model_fn = partial(resnet50, dropout=dropout)
        else:
            from model.base_network import resnet50_mdl
            if args['if_pretrained'] and dataset_name is not None:
                model_fn = partial(resnet50_mdl, dropout=dropout,pretrained=args['if_pretrained'],pretrained_model_path=base_network_path)
            else:
                model_fn = partial(resnet50_mdl, dropout=dropout)
    else:
        from model.base_network import resnet18_mdl_aug
        if args['if_pretrained'] and dataset_name is not None:
            model_fn = partial(resnet18_mdl_aug, dropout=dropout, pretrained=args['if_pretrained'], pretrained_model_path=base_network_path)
        else:
            model_fn  = partial(resnet18_mdl_aug, dropout=dropout)
    model = model_fn(num_classes=num_classes)
    model.to(device)
    return model

def get_optimizer(model, args, params=None):
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    optimizer = args['optimizer']
    params = model.parameters() if params is None else params
    if optimizer=='adam':
        model_optimizer=torch.optim.Adam(params, lr=learning_rate,weight_decay=weight_decay)
    elif optimizer=='momentum':
        model_optimizer=torch.optim.SGD(params, lr=learning_rate,
                                        momentum=0.9,nesterov=True,
                                        weight_decay=weight_decay)
    elif optimizer=='ada':
        model_optimizer=torch.optim.Adadelta(params, lr=learning_rate)
    else:
        raise RuntimeError('No optimizer has been setted!')
    return model_optimizer

def multi_feature_extractor(trainsets, dataset_models, args, num_classes=None):
    extractors = dict()
    temp_pretrained_mode=args['pretrained_mode']
    temp_if_augment = args['if_augment']
    for dataset_name in trainsets:
        if dataset_name not in dataset_models:
            continue
        args['pretrained_mode']='SDL'
        args['if_augment']=False
        if num_classes is None:
            extractor=get_initial_model(None, args, dataset_name)
        else:
            extractor=get_initial_model(num_classes[dataset_name],args, dataset_name)
        ckpt=CheckPointer(args, extractor,optimizer=None)
        args['pretrained_mode'] = temp_pretrained_mode
        args['if_augment']=temp_if_augment
        extractor.eval()
        ckpt.load_model(mode='best', if_initial=True,dataset_name=dataset_name)
        extractors[dataset_name]=extractor
    def multi_embedding(images, return_type='dict', kd=False, logits=False):
        with torch.no_grad():
            all_features=dict()
            all_logits=dict()
            for name, extractor in extractors.items():
                if logits:
                    if kd:
                        all_logits[name],all_features[name]=extractor(images[name],kd=True)
                    else:
                        all_logits[name]=extractor(images[name])
                else:
                    if kd:
                        all_features[name]=extractor.embed(images[name])
                    else:
                        all_features[name]=extractor.embed(images)
        if return_type=='list':
            return list(all_features.values()), list(all_logits.values())
        else:
            return all_features, all_logits
    return multi_embedding
