import torch
# import math,time
import wandb

from torch.utils.data import DataLoader
from model.fcos import FCOSDetector
from data.voc import VOCDataset
from engine.utils import sort_by_score, eval_ap_2d
# from torch.utils.tensorboard import SummaryWriter

# import os
import random

import numpy as np
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from tqdm.auto import tqdm
# import glob

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = dict(
    epochs=30,
    batch_size=8,
    warmup_steps_ratio = 0.12,
    num_workers = 4,
    lr_init=5e-5,
    lr_end=1e-6,
    dataset="PASCALVOC_2012",
    data_path='/home/bruno-messias/Github/data3/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',
    resize_size=[512,800],
    transform = None,
    difficult=False,
    is_train=True,
    architecture="MyFCOS")

def make(config):
    # Make the data
    train_dataset=VOCDataset(config.data_path, 
                             resize_size=config.resize_size, 
                             split='train', 
                             use_difficult=config.difficult,
                             is_train=config.is_train,
                             augment=config.transform)
    
    val_dataset=VOCDataset(config.data_path, 
                           resize_size=config.resize_size,
                           split='val',
                           use_difficult=config.difficult, 
                           is_train=False,
                           augment=None)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size, 
                              shuffle=True, 
                              collate_fn=train_dataset.collate_fn, 
                              num_workers=config.num_workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size, 
                            shuffle=True, 
                            collate_fn=val_dataset.collate_fn, 
                            num_workers=config.num_workers)
    
    print("Total Images [TRAIN] : {}".format(len(train_dataset)))
    print("Total Images [VAL] : {}".format(len(val_dataset)))

    # Make the model
    model = FCOSDetector(mode="inference").to(device)
    model = torch.nn.DataParallel(model)

    # Make the  optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    return model, train_loader, val_loader, optimizer

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="ResearchOD", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, _, val_loader, __ = make(config)

      # save models
      test(model, val_loader)

def test(model, loader):
    # weight_path = './checkpoints' 
    # extension = '.pth' 

    # files = glob.glob(os.path.join(weight_path, '*' + extension))

    # losses = []
    # file_name = []
    # for file in files:
    #     filename = file.split("/")
    #     loss_data = filename[2].split("_")
    #     num = loss_data[3].split("loss")
    #     numbers = num[1].split(".")
    #     loss = f"{numbers[0]}.{numbers[1]}"
    #     losses.append(loss)
    #     file_name.append(file)

    # min_value = min(losses)
    # index_min = losses.index(min_value)

    checkpoint = "./checkpoints/voc2012_512x800_epoch1_loss4.6031.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model=model.to(device).eval()
    print("===>success loading model")

    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    # num=0
    for img,boxes,classes in tqdm(loader):
        with torch.no_grad():
            out=model(img.cuda())
            pred_boxes.append(out[2][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_scores.append(out[0][0].cpu().numpy())
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        # num+=1
        # print(num,end='\r')

    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(loader.CLASSES_NAME))
    print("all classes AP=====>\n")
    for key,value in all_AP.items():
        print('ap for {} is {}'.format(loader.id2name[int(key)],value))
    mAP=0.
    for _,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(loader.CLASSES_NAME)-1)
    print("mAP=====>%.3f\n"%mAP)

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, img, "model.onnx")
    # wandb.save("model.onnx")

if __name__=="__main__":

    model_pipeline(config)