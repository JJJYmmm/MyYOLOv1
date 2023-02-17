import torch
import numpy as np
from torch import nn

from .basic import Conv,SPP
from backbone import build_resnet

from .loss import compute_loss

class myYOLO(nn.Module):
    def __init__(self,device,input_size=None,num_classes=20,trainable=False,conf_thresh=0.01,nms_thresh=0.5):
        super(myYOLO,self).__init__()
        self.device = device            # cuda or cpu
        self.num_classes = num_classes  # 20 or 80
        self.trainable = trainable      # 训练时为true 否则为false
        self.conf_thresh = conf_thresh  # 对最终的检测框进行筛选时所用到的阈值
        self.nms_thresh = nms_thresh    # nms操作需要用到的阈值
        self.stride = 32                # 网络最大降采样倍数
        self.grid_cell = self.create_grid(input_size) # 用于得到最终bbox的参数
        self.input_size = input_size

        # >>>>>>>>>>>>>>>>>>>>>>>>> backbone网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的backbone网络
        self.backbone, feat_dim = build_resnet('resnet18',pretrained=trainable)

        # >>>>>>>>>>>>>>>>>>>>>>>>> neck网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的neck网络
        self.neck = nn.Sequential(
            SPP(),                          # 经过SPP通道数拓展到四倍
            Conv(feat_dim*4,feat_dim,k=1),  # 1X1卷积恢复通道数
        )

        # >>>>>>>>>>>>>>>>>>>>>>>>> detection head网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的head网络
        self.convsets = nn.Sequential(
            Conv(feat_dim,feat_dim//2,k=1),
            Conv(feat_dim//2,feat_dim,k=3,p=1),
            Conv(feat_dim,feat_dim//2,k=1),
            Conv(feat_dim//2,feat_dim,k=3,p=1),
        )
        
        # >>>>>>>>>>>>>>>>>>>>>>>>> 预测层 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.pred = nn.Conv2d(feat_dim, 1 + self.num_classes + 4, 1) # input_size = feat_dim ouput_size = 1 + num_classes + 4 kenerl_size = 1

        if self.trainable:
            self.init_bias()

    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.-init_prob)/init_prob))
        nn.init.constant_(self.pred.bias[...,:1],bias_value)
        nn.init.constant_(self.pred.bias[...,1:1+self.num_classes],bias_value)


    def create_grid(self,input_size):
        # To do：
        # 生成一个tensor：grid_xy，每个位置的元素是网格的坐标，
        # 这一tensor将在获得边界框参数的时候会用到。
        # w and h
        w, h = input_size,input_size
        # 特征图的宽和高
        ws, hs = w // self.stride, h // self.stride
        # 生成网格的x y坐标
        grid_y , grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        # 将xy两部分坐标拼起来 [H,W,2]
        grid_xy = torch.stack([grid_x,grid_y],dim=-1).float()
        # [H,W,2] -> [HW,2]
        grid_xy = grid_xy.view(-1,2).to(self.device)
        
        return grid_xy

    def set_grid(self,input_size):
        # TO do:
        # 重置grid_xy
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)

    # bbox坐标转换见https://zhuanlan.zhihu.com/p/365805593
    def decode_boxes(self,pred):
        # 将网络输出的tx，ty，tw，th四个量转换成bbox的(x1,y1)(x2,y2)
        output = torch.zeros_like(pred)
        # 得到所有bbox 的中心点坐标和宽高
        pred[...,:2] = torch.sigmoid(pred[...,:2]) + self.grid_cell
        pred[...,2:] = torch.exp(pred[...,2:])

        # 将所有bbox的中心坐标和宽高换算成x1y1x2y2形式
        output[...,:2] = pred[...,:2] * self.stride - pred[...,2:] * 0.5
        output[..., 2:] = pred[..., :2] * self.stride + pred[..., 2:] * 0.5

        return output


    def nms(self,dets,scores):
        # 最基本的nms
        # copy from Faster RCNN
        '''Pure Python NMS baseline.''' 
        x1 = dets[:,0] #xmin
        y1 = dets[:,1] #ymin
        x2 = dets[:,2] #xmax
        y2 = dets[:,3] #ymax

        areas = (x2-x1)*(y2-y1)     # bbox的宽w和高h
        order = scores.argsort()[::-1] # 降序排列

        keep = []                   # 保存结果 
        while order.size > 0:
            i = order[0]            # 得到分数最高的box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28,xx2-xx1)
            h = np.maximum(1e-28,yy2-yy1)
            inter = w * h            # 交集大小

            # cross Area/(bbox + paticular area - corss Area)
            iou = inter / (areas[i]+areas[order[1:]] - inter)
            # reserve all the bouding box whose ovr less than thresh
            inds = np.where(iou<=self.nms_thresh)[0]
            order = order[inds + 1]
        
        return keep
        
    def postprocess(self,bboxes,scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """
        labels = np.argmax(scores,axis=1)
        scores = scores[(np.arange(scores.shape[0]),labels)]

        # threshold
        keep = np.where(scores>=self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms
        keep = np.zeros(len(bboxes),dtype=np.int)
        for i in range(self.num_classes):
            # 选择label值为i的cell的下标
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes,c_scores)
            keep[inds[c_keep]] = 1  # 留下合格的bbox

        keep = np.where(keep>0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes,scores,labels

    @torch.no_grad()
    def inference(self,x):
        # backbone
        feat = self.backbone(x)

        # neck
        feat = self.neck(feat)

        # detection head
        feat = self.convsets(feat)

        # pred
        pred = self.pred(feat)

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
        # [B, H*W, 1]
        conf_pred = pred[..., :1]
        # [B, H*W, num_cls]
        cls_pred = pred[..., 1:1+self.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = pred[..., 1+self.num_classes:]

        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        conf_pred = conf_pred[0]            #[H*W, 1]
        cls_pred = cls_pred[0]              #[H*W, NC]
        txtytwth_pred = txtytwth_pred[0]    #[H*W, 4]

        # 每个bbox的得分
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred,dim=-1)

        # 还原bbox坐标 并归一化 [H*W,4]
        bboxes = self.decode_boxes(txtytwth_pred) / self.input_size
        bboxes = torch.clamp(bboxes,0.,1.)

        # move to cpu for postprocessing
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # postprocess
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes,scores,labels


    def forward(self,x,targets=None):
        # 前向推理的代码，主要分为两部分：
        # 训练部分：网络得到obj、cls和txtytwth三个分支的预测，然后计算loss；
        # 推理部分：输出经过后处理得到的bbox、cls和每个bbox的预测得分。           
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone
            feat = self.backbone(x)

            # neck
            feat = self.neck(feat)

            # detection head
            feat = self.convsets(feat)

            # 预测层
            pred = self.pred(feat)

            # 对pred 的size做一些view调整，便于后续的处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            pred = pred.permute(0,2,3,1).contiguous().flatten(1,2)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
            # [B, H*W, 1]
            conf_pred = pred[...,:1]
            # [B, H*W, num_cls]
            cls_pred = pred[..., 1:1+self.num_classes]
            # [B, H*W, 4] 只有一个bbox
            txtytwth_pred = pred[..., 1+self.num_classes:]

            # compute loss
            (   
                conf_loss,
                cls_loss,
                bbox_loss,
                total_loss
            ) = compute_loss(
                pred_conf = conf_pred,
                pred_cls = cls_pred,
                pred_txtytwth = txtytwth_pred,
                targets = targets
            )

            return conf_loss, cls_loss, bbox_loss, total_loss