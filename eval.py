import torch

from data.voc import VOCDataset
from model.fcos import FCOSDetector
from engine.utils import sort_by_score, eval_ap_2d
# from demo import convertSyncBNtoBN

eval_dataset=VOCDataset("/home/data/voc2007_2012/VOCdevkit/VOC2012",resize_size=[800,1024],split='val2007')
print("INFO===>eval dataset has %d imgs"%len(eval_dataset))
eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=eval_dataset.collate_fn)

model=FCOSDetector(mode="inference")
# model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
# print("INFO===>success convert BN to SyncBN")
model.load_state_dict(torch.load("./logs/voc20172012_multigpu_800x1024_epoch27_loss0.5987.pth", map_location=torch.device('cpu')))
# model=convertSyncBNtoBN(model)
# print("INFO===>success convert SyncBN to BN")
model=model.cuda().eval()
print("===>success loading model")

gt_boxes=[]
gt_classes=[]
pred_boxes=[]
pred_classes=[]
pred_scores=[]
num=0
for img,boxes,classes in eval_loader:
    with torch.no_grad():
        out=model(img.cuda())
        pred_boxes.append(out[2][0].cpu().numpy())
        pred_classes.append(out[1][0].cpu().numpy())
        pred_scores.append(out[0][0].cpu().numpy())
    gt_boxes.append(boxes[0].numpy())
    gt_classes.append(classes[0].numpy())
    num+=1
    print(num,end='\r')

# print(gt_boxes[0],gt_classes[0])
# print(pred_boxes[0],pred_classes[0],pred_scores[0])

pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(eval_dataset.CLASSES_NAME)+1)
print("all classes AP=====>\n",all_AP)
mAP=0.
for class_id,class_mAP in all_AP.items():
    mAP+=float(class_mAP)
mAP/=(len(eval_dataset.CLASSES_NAME)+1)
print("mAP=====>%.3f\n"%mAP)
