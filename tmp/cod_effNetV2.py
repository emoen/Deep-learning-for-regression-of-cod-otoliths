#!/usr/bin/env python
# coding: utf-8

# In[1]:
#apt-get update
#apt-get -qq -y install curl
#apt install libgl1-mesa-glx
#pip uninstall opencv-python-headless
#pip install opencv-python-headless==4.1.2.30


import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from PIL import Image, ExifTags

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, accuracy_score
import scipy

import matplotlib.pyplot as plt

import os
import gc
import copy
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from tqdm import tqdm
from collections import defaultdict

from loguru import logger

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from colorama import Fore
b_ = Fore.BLUE

from train_val_test_split import train_validate_test_split


# ### Train Configuration

# In[3]:


class CONFIG:
    seed = 42
    model_name = 'tf_efficientnetv2_l_in21k'  #V2-l with all weights
    train_batch_size = 8
    valid_batch_size = 8
    img_size = 384
    val_img_size = 384 #480
    learning_rate = 1e-5
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = 10
    scheduler = 'CosineAnnealingLR'
    n_accumulate = 1
    n_fold = 10 #5
    target_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    debugging = False
    which_exposure =  "all" #"middle" #{min, middle, max}
    in_chans = 9 # number of channels for model - either 3 (RGB) or 9 - 3 images
    CHANNELS = "channels_first"
    KERAS_TRAIN_TEST_SEED = 2021
    #ROOTDIR = "./EFFNetV2_l_MLP/"
    #ROOTDIR = "./REXNET_middle/"
    #ROOTDIR = "./EFFNetV2_l_MLP_all/"
    ROOTDIR = "./EFFNetV2_l_all_mse/"
    CUDA_VISIBLE_DEVICE = "0"
    #tensorboard_path = 'tensorboard_test2'
    #checkpoint_path = 'checkpoints_test2/cod_oto_efficientnetBBB.{epoch:03d}-{val_loss:.2f}.hdf5'
    input_shape = (3, img_size, img_size)
    test_size = 0.1 #0.15
    test_split_seed = 8
    steps_per_epoch = 1600
    epochs = 450
    early_stopping_patience = 14
    reduceLROnPlateau_factor = 0.2
    reduceLROnPlateau_patience = 7
    early_stopping = 40
    base_dir = '/gpfs/gpfs0/deep/data/Savannah_Professional_Practice2021_08_12_2021/CodOtholiths-MachineLearning/Savannah_Professional_Practice'
    
    
def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG.seed)    


# In[4]:


import json

config_dict = CONFIG.__dict__
config_dict = dict(config_dict)
config_dict.pop('device', None)
config_dict.pop('__dict__', None)
config_dict.pop('__weakref__', None)
config_dict.pop('__doc__', None)

with open(CONFIG.ROOTDIR+'config.json', 'w', encoding='utf-8') as f:
    json.dump(config_dict, f, ensure_ascii=False, indent=4)


# In[5]:


from utils.read_jpg_cods import read_jpg_cods
from utils.train_val_test_split import *
from utils.train_test_split import *

df = read_jpg_cods( CONFIG ) 


# In[6]:


CONFIG.img_size = CONFIG.val_img_size
df_test = read_jpg_cods( CONFIG ) 


# ### Train/Test split

# In[7]:


print("len age:"+str( len(df.age) ) ) #len age:5090, error_count:205
train_imgs, train_age, test_imgs, test_age, test_path = train_test_split(df, CONFIG, test_size=CONFIG.test_size, a_seed=CONFIG.test_split_seed)
test_path.to_csv( CONFIG.ROOTDIR+"test_set_files.csv", index=False)
train_imgs2, train_age2, test_imgs2, test_age2, test_path2 = train_test_split(df_test, CONFIG, test_size=CONFIG.test_size, a_seed=CONFIG.test_split_seed)

print(np.any(test_path2==test_path))
print(np.any(test_age==test_age2))
test_imgs = test_imgs2

del train_imgs2
del test_imgs2
del train_age2
del test_age2
print(test_imgs.shape)

#df1 = pd.DataFrame(list(zip(train_imgs, train_age)), columns=['image', 'age'])

print(train_imgs.shape)
print(len(train_age))
print(test_imgs.shape)
print(len(test_age))
print(len(train_age)+len(test_age))
print(len(df))
#print(len(df1))


# ### Dataset class

# In[8]:


class codDataset(Dataset):
    def __init__(self, imgs, age, transform=None): 
        self.labels = age 
        self.image = imgs #np.stack( df['image'].values , axis=0) # make 4D-array (num_imgs, channels, width, height)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.image[index]
        label = torch.tensor(self.labels[index]).float()
        
        if self.transform:
            #print("1"+str(image.shape)) 
            tmp = image.transpose(1,2,0) # convert to channel last
            #print("2"+str(tmp.shape))
            image = self.transform(image=tmp)["image"] 
            #print("3"+str(image.shape))
            image = image.transpose(2,0,1)
            #print("4"+str(image.shape))
        
        return image, label


# ### Augmentation

# #### This one

# In[9]:


data_transforms = {
    "train": A.Compose([A.Rotate((-90, 90), p=1.0)], p=0.99),
    "valid": A.Compose([], p=1.)
}


# ### Cod Model

# In[10]:


class codModel(nn.Module):

    def __init__(self, model_name, pretrained=True):
        super(codModel, self).__init__()
        self.model = timm.create_model(CONFIG.model_name, pretrained=pretrained, in_chans=CONFIG.in_chans, num_classes=1) #model_name
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, CONFIG.target_size)
        lastLayer = nn.Sequential(nn.Linear(self.n_features, 256),
              nn.LeakyReLU(),
              nn.Linear(256, 32),
              nn.LeakyReLU(),
              nn.Linear(32, CONFIG.target_size))
        self.model.classifier = lastLayer
        print("model self:"+str(self.model.classifier))

    def forward(self, x):
        output = self.model(x)
        return output
    
model = codModel(CONFIG.model_name)
model.to(CONFIG.device);


# In[20]:


#model.eval()


# ### Training function

# In[11]:


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    optimizer.zero_grad()
    loss_fn = nn.MSELoss()
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:  
        #optimizer.zero_grad()
        
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        labels = torch.unsqueeze(labels, 1)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            outputs = model(images)
            #print("outputs:+"+str(outputs))
            #print("labels:"+str(labels))
            #print("mse:"+str(mean_squared_error(labels.cpu().data.numpy(), outputs.cpu().data.numpy())))
            loss = loss_fn(outputs, labels)
         
        
        scaler.scale(loss).backward() # Scales loss.  Calls backward() on scaled loss to create scaled gradients
        #model.print_debug() #model.classifier.weight[0:10,0]
        
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.        
        scaler.step(optimizer)
        scaler.update() # Updates the scale for next iteration.
            
        # zero the parameter gradients
        optimizer.zero_grad() # set_to_none=True here can modestly improve performance
                
        running_loss += loss.item() #(loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss/dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss


# ### Validation function

# In[12]:


from sklearn.metrics import accuracy_score, mean_squared_error

@torch.no_grad()
def valid_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    TARGETS = []
    PREDS = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar: 
        images = images.to(device)
        labels = labels.to(device)
        labels = torch.unsqueeze(labels, 1)
        outputs = model(images)
        loss = nn.MSELoss()(outputs, labels)
        
        batch_size = images.size(0)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss/dataset_size
        
        PREDS.append(outputs.cpu().detach().numpy())
        TARGETS.append(labels.view(-1).cpu().detach().numpy())
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
    
    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    PREDS = np.squeeze(PREDS)
    
    print("max:"+str(np.max( PREDS )))
    print("mean:"+str(np.mean( PREDS )))
    print("min:"+str(np.min( PREDS )))
    
    PREDS = PREDS.round()
    val_auc = accuracy_score(TARGETS, PREDS) #roc_auc_score(TARGETS, PREDS)
    mse_score = mean_squared_error(TARGETS, PREDS)
    print("acc:"+str( val_auc ) )
    print("mse:"+str( mse_score ) )
    gc.collect()
    
    return epoch_loss , val_auc, mse_score


# ### Run

# In[13]:


@logger.catch
def run(model, optimizer, scheduler, train_loader, valid_loader, fold_i, test_imgs):   
    device=CONFIG.device
    num_epochs=CONFIG.epochs
    patience = CONFIG.early_stopping
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    #best_epoch_auc = 0
    best_epoch_auc = 1.0
    best_epoch = 0
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG.device, epoch=epoch)
        
        valid_epoch_loss, acc_score, mse_score = valid_one_epoch(model, optimizer, scheduler,
                                                            dataloader=valid_loader, 
                                                            device=CONFIG.device, epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(valid_epoch_loss)
        
        #history['Valid AUC'].append(acc_score) #valid_epoch_auc)
        history['Valid AUC'].append(mse_score) #valid_epoch_auc)
        
        #print(f'Valid AUC: {valid_epoch_auc}')
        
        if scheduler is not None:
            scheduler.step()
        
        ############## predict on test-set ################
        PREDS_TEST  = []
        ceil_step = int(test_imgs.shape[0] / 2) + int(test_imgs.shape[0] % 2 > 0)
        for i in range(0, ceil_step):
            start_step = i * 2
            preds = model( torch.Tensor(test_imgs[start_step:start_step+2]).to(CONFIG.device) )
            preds = preds.cpu().detach().numpy()
            preds = preds.flatten()
            PREDS_TEST.extend(preds.tolist())
            
        print("max:"+str(np.max( PREDS_TEST )))
        print("mean:"+str(np.mean( PREDS_TEST )))
        print("min:"+str(np.min( PREDS_TEST )))
        ROUNDED_PREDS_TEST = np.asarray(PREDS_TEST).round().astype('int')
        print("test mse:"+str( mean_squared_error(PREDS_TEST, test_age) )) 
        print("test acc:"+str( accuracy_score(ROUNDED_PREDS_TEST.tolist(), test_age.tolist() ) )) 
        print("PREDS TYPE:"+str(type(PREDS_TEST)))     
        ############## END: predict on test-set ################            
        
        # deep copy the model
        #if acc_score >= best_epoch_auc:
        if mse_score <= best_epoch_auc:            
            #print(f"{b_}Validation AUC Improved ({best_epoch_auc} ---> {acc_score})")
            #best_epoch_auc = acc_score
            print(f"{b_}Validation AUC Improved ({best_epoch_auc} ---> {mse_score})")
            best_epoch_auc = mse_score
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = CONFIG.ROOTDIR +"AUC{:.4f}_epoch{:.0f}_fold_{:.0f}.bin".format(best_epoch_auc, epoch, fold_i)
            torch.save(model.state_dict(), PATH)
            print("Model Saved")
            
        print()
        if best_epoch < epoch - patience and epoch > 150:
            break
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUC: {:.4f}".format(best_epoch_auc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    #print(model.classifier.weight[0:10,0])
    
    return model, history, best_model_wts


# ### Train fold 0 to 5

# #CONFIG.n_fold
# CONFIG.epochs

# In[ ]:


from sklearn.metrics import accuracy_score, mean_squared_error

plt.rcParams['figure.dpi'] = 150

def fetch_scheduler(optimizer):
    if CONFIG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.T_max, eta_min=CONFIG.min_lr)
    elif CONFIG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG.T_0, T_mult=1, eta_min=CONFIG.min_lr)
    elif CONFIG.scheduler == None:
        return None
    
        
    return scheduler

optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
scheduler = fetch_scheduler(optimizer)

print(CONFIG.device)
print(CONFIG.epochs)
#test_img = np.stack(df_test['image'].values  , axis=0)
#test_img = test_img * 1.0/255.0
print(type(test_imgs))
print(test_imgs.shape)
print(type(test_age))
print("#########")
print(type(train_imgs))
print(train_imgs.shape)

test_img = torch.from_numpy(test_imgs)
print("test_img shape:"+str( test_imgs.shape) )
print("test_img[0].shape:"+str( test_imgs[0].shape))
#test_img.to(CONFIG.device)
print("max test_imgs0 value:"+str( np.max(test_imgs[0]) ) )
print("min test_imgs0 value:"+str( np.min(test_imgs[0]) ) )

a_seed = CONFIG.KERAS_TRAIN_TEST_SEED #2021
numberOfFolds = CONFIG.n_fold #5
kfold = StratifiedKFold(n_splits=numberOfFolds, random_state=a_seed, shuffle=True)
#for i in range(0,5):
#CONFIG.epochs=5
test_15 = pd.DataFrame()
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_imgs, train_age.tolist())):
    #train_loader, valid_loader = prepare_data(fold=i)
    ######## K-FOLD Start #################
    train_imgs_new = train_imgs[train_idx]
    train_age_new = train_age[train_idx]
    val_imgs_new = train_imgs[val_idx]
    val_age_new = train_age[val_idx]
    
    train_dataset = codDataset(train_imgs_new, train_age_new, data_transforms["train"])
    valid_dataset = codDataset(val_imgs_new, val_age_new, data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.train_batch_size, 
                              num_workers=0, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.valid_batch_size, 
                              num_workers=0, shuffle=False, pin_memory=True)
    ######################### K-FOLD End #########################
    model, history, best_model_wts = run(model, optimizer, scheduler, train_loader, valid_loader, fold, test_imgs)

    fig = plt.figure(figsize=(5, 5))
    with torch.no_grad():
        PREDS_TEST  = []
        
        ceil_step = int(test_imgs.shape[0] / 25) + int(test_imgs.shape[0] % 25 > 0)
        for i in range(0, ceil_step):
            start_step = i * 25
            preds = model( torch.Tensor(test_imgs[start_step:start_step+25]).to(CONFIG.device) )
            preds = preds.cpu().detach().numpy()
            preds = preds.flatten()
            PREDS_TEST.extend(preds.tolist())
            
            if i == 0 :
                fig = plt.figure(figsize=(5, 5))
                plt.subplots_adjust(hspace = 2.0)
                #test_imgs.shape[0]
                for j in range(0, 25):
                    fig.add_subplot(5, 5, j+1)
                    
                    tmp = test_imgs[j]
                    channel1 = tmp[0:9:3,:,:]
                    channel2 = tmp[1:9:3,:,:]
                    channel3 = tmp[2:9:3,:,:]
                    channel1 = np.mean(channel1, axis=0 )
                    channel2 = np.mean(channel2, axis=0 )
                    channel3 = np.mean(channel3, axis=0 )
                    d3_img = np.stack((channel1, channel2, channel3), axis=0)
                    plt.imshow( d3_img.transpose(1,2,0) )
                    
                    #plt.imshow( test_imgs[j].transpose(1,2,0) )
                    plt.title(f'{preds[j]:.2f} ,{test_age[j]}')
                    plt.xticks([])
                    plt.yticks([])
                plt.tight_layout()
                plt.show()

        print("max:"+str(np.max( PREDS_TEST )))
        print("mean:"+str(np.mean( PREDS_TEST )))
        print("min:"+str(np.min( PREDS_TEST )))

        ROUNDED_PREDS_TEST = np.asarray(PREDS_TEST).round().astype('int')
        #print(ROUNDED_PREDS_TEST ) 
        #print(type(ROUNDED_PREDS_TEST))
        #print(ROUNDED_PREDS_TEST.shape)

        print("test mse:"+str( mean_squared_error(PREDS_TEST, test_age) )) 
        print("test acc:"+str( accuracy_score(ROUNDED_PREDS_TEST.tolist(), test_age.tolist() ) )) 
        print("PREDS TYPE:"+str(type(PREDS_TEST)))
        #print(PREDS_TEST.shape) ##list
        #print("PREDS:"+str(ROUNDED_PREDS_TEST))
        #print("TRUE:"+str(test_age))

        del model
        model = codModel(CONFIG.model_name)
        model.to(CONFIG.device);
        
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
        scheduler = fetch_scheduler(optimizer)
        test_15[str(fold)] = PREDS_TEST
        test_15.to_csv(CONFIG.ROOTDIR+'preds_'+str(fold)+'.csv', index=False)
        
test_15.to_csv(CONFIG.ROOTDIR+'preds.csv', index=False)

