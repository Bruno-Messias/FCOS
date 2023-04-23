import torch
import numpy as np
import cv2
from engine.utils import sort_by_score, iou_2d, _compute_ap, eval_ap_2d

if __name__=="__main__":
    from model.fcos import FCOSDetector
    from demo import convertSyncBNtoBN
    from dataset.VOC_dataset import VOCDataset
    

    eval_dataset = VOCDataset(root_dir='/Users/zhangzhenghao/VOC0712', resize_size=[800, 1333],
                               split='test', use_difficult=False, is_train=False, augment=None)
    print("INFO===>eval dataset has %d imgs"%len(eval_dataset))
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=eval_dataset.collate_fn)

    model=FCOSDetector(mode="inference")
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/voc_78.7.pth",map_location=torch.device('cpu')))
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

    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(eval_dataset.CLASSES_NAME))
    print("all classes AP=====>\n")
    for key,value in all_AP.items():
        print('ap for {} is {}'.format(eval_dataset.id2name[int(key)],value))
    mAP=0.
    for class_id,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(eval_dataset.CLASSES_NAME)-1)
    print("mAP=====>%.3f\n"%mAP)