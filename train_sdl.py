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
from model.losses import cross_entropy_loss, prototype_loss, distillation_loss, DistillKL
from model.model_utils import UniformStepLR, CosineAnnealRestartLR, ExpDecayLR, WeightAnnealing
from model.get_model import get_initial_model, get_optimizer, multi_feature_extractor, CheckPointer
from model.pa import apply_selection, pa
from model.base_network import FT
from config import device, args, BATCHSIZES, LOSSWEIGHTS, KDFLOSSWEIGHTS, KDPLOSSWEIGHTS, KDANNEALING, DATASET_MODELS_DICT

trainsets, valsets, testsets = args['train_datasets'], args['valid_datasets'], args['test_datasets']

train_loaders = []
num_train_classes = {}
kd_weight_annealing = {}

for t_index, trainset in enumerate(trainsets):
    train_loaders.append(MetaDatasetBatchReader('train', [trainset], valsets, testsets, batch_size=BATCHSIZES[trainset]))
    num_train_classes[trainset] = train_loaders[t_index].num_classes('train')
    kd_weight_annealing[trainset] = WeightAnnealing(T=int(args['valid_frequency']*KDANNEALING[trainset]))

val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)

#每个域作为分类类别
model = get_initial_model(len(trainsets), args, None)

optimizer = get_optimizer(model, args, params=model.get_parameters())

extractor_domains = trainsets
dataset_models = DATASET_MODELS_DICT[args['pretrained_model_name']]
multi_embedding = multi_feature_extractor(extractor_domains, dataset_models, args, num_train_classes)

checkpointer = CheckPointer(args, model, optimizer=optimizer)
if os.path.isfile(checkpointer.temp_load_path+'/'+checkpointer.temp_model_name) and args['if_recover_from_log']:
    start_epoch, best_val_loss, best_val_acc = checkpointer.load_model()
else:
    print('there is no restoration checkpoint, initialize...')
    best_val_loss = 99999
    start_epoch = 0
    best_val_acc = 0

if args['lr_policy'] == "step":
    lr_manager = UniformStepLR(optimizer, args, start_epoch)
elif args['lr_policy'] == "exp":
    lr_manager = ExpDecayLR(optimizer, args, start_epoch)
elif args['lr_policy'] == "cosine":
    lr_manager = CosineAnnealRestartLR(optimizer, args, start_epoch)

summary_out_path=args['result_outputs']+'/'+args['train_mode']+'_'+args['pretrained_model_name']
writer = SummaryWriter(summary_out_path)
max_epoch = args['epochs']
epoch_loss = {name: [] for name in trainsets}
epoch_kd_f_loss = {name: [] for name in trainsets}
epoch_kd_p_loss = {name: [] for name in trainsets}
epoch_acc = {name: [] for name in trainsets}

epoch_val_loss = {name: [] for name in valsets}
epoch_val_acc = {name: [] for name in valsets}

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = False
with tf.compat.v1.Session(config=tf_config) as session:
    for i in tqdm(range(max_epoch)):
        if i<start_epoch:
            continue

        optimizer.zero_grad()
        optimizer_adaptor.zero_grad()

        samples = []
        images = dict()
        num_samples = []
        for t_index, (name, train_loader) in enumerate(zip(trainsets, train_loaders)):
            sample = train_loader.get_train_batch(session)
            samples.append(sample)
            images[name] = sample['images']
            num_samples.append(sample['images'].size(0))
        if torch.isnan(torch.cat(list(images.values()))).any():
            raise RuntimeError("image have nan values!!!")
        #print('num_samples:',num_samples)
        mtl_logits, mtl_features = model.forward(torch.cat(list(images.values()),dim=0), num_samples, kd=True)
        stl_features, stl_logits = multi_embedding(images, return_type='list', kd=True, logits=True)
        mtl_features = A(mtl_features)

        batch_losses, state_dicts = [], []
        kd_f_losses = 0
        kd_p_losses = 0
        for t_index, trainset in enumerate(trainsets):
            #print('t_index:',t_index,trainset)
            if torch.isnan(mtl_logits[t_index]).any():
                raise RuntimeError("mtl_logits_",t_index, " have nan values")
            if torch.isnan(mtl_features[t_index]).any():
                raise RuntimeError("mtl_features_",t_index, " have nan values")
            if torch.isnan(stl_logits[t_index]).any():
                raise RuntimeError("stl_logits_",t_index, " have nan values")
            if torch.isnan(stl_features[t_index]).any():
                raise RuntimeError("stl_features_",t_index, " have nan values")
            batch_loss, state_dict, _ = cross_entropy_loss(mtl_logits[t_index], samples[t_index]['labels'])
            #print("batch_loss:",batch_loss)
            #print("epoch_loss:",state_dict['loss'])
            #print("epoch_acc:",state_dict['acc'])
            batch_losses.append(batch_loss*LOSSWEIGHTS[trainset])
            state_dicts.append(state_dict)

            batch_dataset = samples[t_index]['dataset_name']
            epoch_loss[batch_dataset].append(state_dict['loss'])
            epoch_acc[batch_dataset].append(state_dict['acc'])
            #print('mtl_logits:',mtl_logits[t_index].shape,'  mtl_feature:',mtl_features[t_index].shape)
            #print('stl_logits:',stl_logits[t_index].shape,'  stl_feature:',stl_features[t_index].shape)
            ft = torch.nn.functional.normalize(stl_features[t_index], p=2, dim=1, eps=1e-12)
            fs = torch.nn.functional.normalize(mtl_features[t_index], p=2, dim=1, eps=1e-12)
            if torch.isnan(ft).any():
                raise RuntimeError('ft have nan values!')
            if torch.isnan(fs).any():
                raise RuntimeError('fs have nan values!')
            kd_f_loss = distillation_loss(fs, ft.detach(), opt='kernelcka')
            kd_p_loss = criterion_div(mtl_logits[t_index], stl_logits[t_index])
            #print("kd_f_loss:",kd_f_loss)
            #print("kd_p_loss:",kd_p_loss)
            kd_weight = kd_weight_annealing[trainset](t=i, opt='linear')*KDFLOSSWEIGHTS[trainset]
            bam_weight = kd_weight_annealing[trainset](t=i, opt='linear')*KDPLOSSWEIGHTS[trainset]
            #if kd_weight>0:
                #kd_f_losses += kd_f_loss*kd_weight
            kd_f_losses += kd_f_loss*kd_weight
            #if bam_weight>0:
                #kd_p_losses += kd_p_loss*bam_weight
            kd_p_losses += kd_p_loss*bam_weight
            epoch_kd_f_loss[batch_dataset].append(kd_f_loss.item())
            epoch_kd_p_loss[batch_dataset].append(kd_p_loss.item())

        batch_loss = torch.stack(batch_losses).sum()
        kd_f_losses = kd_f_losses*args['KL_f_weight']
        kd_p_losses = kd_p_losses*args['KL_p_weight']
        batch_loss = batch_loss+kd_p_losses+kd_f_losses

        batch_loss.backward()
        for name, parms in model.named_parameters():
            #print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
            if torch.isnan(parms.grad).any():
                raise RuntimeError(name," grad have nan values!!!")

        optimizer.step()
        optimizer_adaptor.step()
        lr_manager.step(i)
        lr_manager_ad.step(i)

        if (i+1)%args['writer_frequency']==0:
            for dataset_name in trainsets:
                writer.add_scalar(f"train_loss/{dataset_name}-train_loss", np.mean(epoch_loss[dataset_name]), i)
                writer.add_scalar(f"train_accuracy/{dataset_name}-train_acc", np.mean(epoch_acc[dataset_name]), i)
                writer.add_scalar(f"kd_f_loss/{dataset_name}-train_kd_f_loss", np.mean(epoch_kd_f_loss[dataset_name]), i)
                writer.add_scalar(f"kd_p_loss/{dataset_name}-train_kd_p_loss", np.mean(epoch_kd_p_loss[dataset_name]), i)
                epoch_loss[dataset_name], epoch_acc[dataset_name], epoch_kd_f_loss[dataset_name], epoch_kd_p_loss[dataset_name] = [], [], [], []
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i)

        if (i+1)%args['valid_frequency']==0:
            model.eval()
            dataset_accs, dataset_losses = [], []
            for valset in valsets:
                val_losses, val_accs = [], []
                for j in tqdm(range(args['valid_size'])):
                    with torch.no_grad():
                        sample = val_loader.get_validation_task(session, valset)
                        context_features = model.embed(sample['context_images'])
                        target_features = model.embed(sample['target_images'])
                        context_labels = sample['context_labels']
                        target_labels = sample['target_labels']
                        _, state_dict, _ = prototype_loss(context_features, context_labels, target_features, target_labels)
                    val_losses.append(state_dict['loss'])
                    val_accs.append(state_dict['acc'])
                dataset_acc, dataset_loss = np.mean(val_losses)*100, np.mean(val_losses)
                dataset_losses.append(dataset_loss)
                dataset_accs.append(dataset_acc)
                epoch_val_acc[valset].append(dataset_acc)
                epoch_val_loss[valset].append(dataset_loss)

                writer.add_scalar(f"val_loss/{valset}/val_loss", dataset_loss, i)
                writer.add_scalar(f"val_accuracy/{valset}/val_acc", dataset_acc, i)
                print(f"{valset}: val_acc {dataset_acc:.2f}%, val_loss {dataset_loss:.3f}")

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
                          'epoch_val_acc': epoch_val_acc,
                          'adaptors': A.state_dict(),
                          'optimizer_adaptor': optimizer_adaptor.state_dict()}
            checkpointer.save_model(is_best=is_best, epoch=i, best_val_acc=best_val_acc, best_val_loss=best_val_loss,
                                    new_model_state=model.get_state_dict(), new_optimizer_state=optimizer.get_state_dict(), extra_info=extra_dict)

            model.train()
            print(f"trained and evaluted at {i}")

writer.close()
if start_epoch<max_epoch:
    print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, best_avg_val_acc: {best_val_acc:.2f}%""")
else:
    print(f"""No training happened. Loaded checkpoint at {start_epoch}, while max_iter was {max_epoch}""")

