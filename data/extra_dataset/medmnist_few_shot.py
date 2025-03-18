from tqdm import tqdm
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import medmnist
from medmnist import INFO, Evaluator

selected_flag=['tissuemnist','organamnist']

data_out_path='your_path/medmnist_to_mnist'
label_index_temp=0
max_num=1000

for meddata in selected_flag:
    data_file='your_path/medmnist/'+meddata+'.npz'
    data_cont=np.load(data_file)

    print(meddata,'files: ',data_cont.files)#['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']
                                             #(num,28,28), (num,1)
    temp_train_labels=data_cont['train_labels']
    label_num=len(set(temp_train_labels.flatten()))#一个数据集中的类数量
    for i in range(label_index_temp, label_index_temp+label_num):
        temp_label_file_path=data_out_path+'/class_'+str(i)
        if not os.path.exists(temp_label_file_path):
            os.makedirs(temp_label_file_path)
    label_index_name = [0] * label_num
    label_index_flag=[1]*label_num

    for i in tqdm(range(temp_train_labels.shape[0]),postfix=meddata+'/train'):
        np_label = data_cont['train_labels'][i][0]
        if sum(label_index_flag)==0:
            break
        if label_index_name[np_label]>=max_num:
            label_index_flag[np_label]=0
            continue
        #print('train:',i)
        np_data=data_cont['train_images'][i]
        #np_image=np.array([np_data])
        #print(np_data.shape)
        file_label=np_label+label_index_temp
        np_image=Image.fromarray(np_data)
        #print(np_image)
        #print(label_index_name[np_label])

        file_name=data_out_path+'/class_'+str(file_label)+'/image_'+str(label_index_name[np_label])+'.png'
        #print('train:', i, file_name)
        #print(file_name)
        #np_image.show()
        np_image.save(file_name)
        label_index_name[np_label]+=1
    for i in tqdm(range(data_cont['val_labels'].shape[0]),postfix=meddata+'/val'):
        #print('val:',i)
        np_label = data_cont['train_labels'][i][0]
        if sum(label_index_flag) == 0:
            break
        if label_index_name[np_label] >= max_num:
            label_index_flag[np_label] = 0
            continue
        np_data=data_cont['val_images'][i]
        #np_image=np.array([np_data])
        #print(np_data.shape)
        #np_label=data_cont['val_labels'][i][0]
        file_label=np_label+label_index_temp
        np_image=Image.fromarray(np_data)
        #print(np_image)
        #print(label_index_name[np_label])
        file_name=data_out_path+'/class_'+str(file_label)+'/image_'+str(label_index_name[np_label])+'.png'
        #print('val:', i, file_name)
        #print(file_name)
        #np_image.show()
        np_image.save(file_name)
        label_index_name[np_label]+=1
    for i in tqdm(range(data_cont['test_labels'].shape[0]),postfix=meddata+'/test'):
        #print('test:',i)
        np_label = data_cont['train_labels'][i][0]
        if sum(label_index_flag) == 0:
            break
        if label_index_name[np_label] >= max_num:
            label_index_flag[np_label] = 0
            continue
        np_data=data_cont['test_images'][i]
        #np_image=np.array([np_data])
        #print(np_data.shape)
        #np_label=data_cont['test_labels'][i][0]
        file_label=np_label+label_index_temp
        np_image=Image.fromarray(np_data)
        #print(np_image)
        #print(label_index_name[np_label])
        file_name=data_out_path+'/class_'+str(file_label)+'/image_'+str(label_index_name[np_label])+'.png'
        #print('test:',i, file_name)
        #print(file_name)
        #np_image.show()
        np_image.save(file_name)
        label_index_name[np_label]+=1
    #break
    label_index_temp+=label_num
