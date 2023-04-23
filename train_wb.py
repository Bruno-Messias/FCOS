import torch
import math,time
import wandb

from torch.utils.data import DataLoader
from model.fcos import FCOSDetector
from data.voc import VOCDataset
from engine.utils import sort_by_score, eval_ap_2d
from torch.utils.tensorboard import SummaryWriter

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import glob

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#! Add augmentation
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
    architecture="MyFCOS")

def make(config):
    # Make the data
    train_dataset=VOCDataset(config.data_path, resize_size=config.resize_size, split='train')
    val_dataset=VOCDataset(config.data_path, resize_size=config.resize_size,split='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size, 
                              shuffle=True, 
                              collate_fn=train_dataset.collate_fn, 
                              num_workers=config.num_workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size, 
                            shuffle=True, 
                            collate_fn=train_dataset.collate_fn, 
                            num_workers=config.num_workers)

    # Make the model
    model = FCOSDetector(mode="training").to(device)
    model = torch.nn.DataParallel(model)

    # Make the  optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    return model, train_loader, val_loader, optimizer

def train(model, loader, optimizer, config):
    model.train()
    global_steps = 1
    steps_per_epoch=len(loader)//config.batch_size
    total_steps=steps_per_epoch*config.epochs
    warmup_steps = total_steps*config.warmup_steps_ratio

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="all", log_freq=10)
    
    for epoch in tqdm(range(config.epochs)):
        for epoch_step, data in enumerate(loader):

            batch_imgs,batch_boxes,batch_classes=data
            batch_imgs=batch_imgs.to(device)
            batch_boxes=batch_boxes.to(device)
            batch_classes=batch_classes.to(device)

            lr = lr_func(config, global_steps, total_steps, warmup_steps)
            for param in optimizer.param_groups: #? Tem alguma implementação no pytorch para essa mudança de learning rate?
                param['lr'] = lr
            
            start_time=time.time()

            # Forward pass ➡
            losses=model([batch_imgs,batch_boxes,batch_classes])
            # loss=losses[-1] 
            loss = sum(losses)
            #? Pq apenas a última loss? Ver o do unbiased? e https://github.com/zhenghao977/FCOS-PyTorch-37.2AP/blob/master/train_voc.py

            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()

            end_time=time.time()
            cost_time=int((end_time-start_time)*1000)

            
             # Report metrics every 25th batch
            if (global_steps % 50) == 0:
                train_log(global_steps, losses, epoch, epoch_step, steps_per_epoch, cost_time, lr)

            global_steps+=1
        
        torch.save(model.state_dict(),"./checkpoints/voc2012_512x800_epoch%d_loss%.4f.pth"%(epoch+1,loss.item()))

def lr_func(config, global_steps, total_steps, warmup_steps):
    if global_steps<warmup_steps:
        lr=global_steps/warmup_steps*config.lr_init
    else:
        lr=config.lr_end+0.5*(config.lr_init-config.lr_end)*(
            (1+math.cos((global_steps-warmup_steps)/(total_steps-warmup_steps)*math.pi))
        )
    return float(lr)

def train_log(global_steps, losses, epoch, epoch_step, steps_per_epoch, cost_time, lr):
    # Where the magic happens
    print("\n global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e"%\
                (global_steps,epoch+1,epoch_step+1,steps_per_epoch,losses[0],losses[1],losses[2],cost_time,lr))

    wandb.log({ "epoch": epoch+1, 
                "cls_loss": losses[0],
                "cnt_loss": losses[1],
                "reg_loss": losses[2],
                "lr": lr},
                step=global_steps)

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="ResearchOD", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, val_loader, optimizer = make(config)
    #   print(model)

      # and use them to train the model
      train(model, train_loader, optimizer, config)

      # save models
    #   test(val_loader)

def test(loader):
    weight_path = './checkpoints' 
    extension = '.pth' 

    files = glob.glob(os.path.join(weight_path, '*' + extension))

    losses = []
    file_name = []
    for file in files:
        filename = file.split("/")
        loss_data = filename[2].split("_")
        num = loss_data[3].split("loss")
        numbers = num[1].split(".")
        loss = f"{numbers[0]}.{numbers[1]}"
        losses.append(loss)
        file_name.append(file)

    min_value = min(losses)
    index_min = losses.index(min_value)

    model=FCOSDetector(mode="inference")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(file_name[index_min], map_location=torch.device('cpu')))
    model=model.to(device).eval()
    print("===>success loading model")

    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    num=0
    for img,boxes,classes in loader:
        with torch.no_grad():
            out=model(img.cuda())
            pred_boxes.append(out[2][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_scores.append(out[0][0].cpu().numpy())
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        num+=1
        print(num,end='\r')

    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)

    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(loader.CLASSES_NAME)+1)
    print("all classes AP=====>\n",all_AP)

    mAP=0.
    for class_id,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(loader.CLASSES_NAME)+1)
    print("mAP=====>%.3f\n"%mAP)

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, img, "model.onnx")
    wandb.save("model.onnx")

if __name__=="__main__":

    folder_checkpoint = './checkpoints' 
    if not os.path.exists(folder_checkpoint):
        os.mkdir(folder_checkpoint)

    model_pipeline(config)