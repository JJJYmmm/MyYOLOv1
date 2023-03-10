import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MSEWithLogitsLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, logits, targets):
        inputs  = torch.clamp(torch.sigmoid(logits),min=1e-4,max=1.0-1e-4)

        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * (inputs-targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        return loss

def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算边界框中心点
    c_x = (xmax+xmin) / 2 * w
    c_y = (ymax+ymin) / 2 * h
    box_w = (xmax-xmin) * w
    box_h = (ymax-ymin) * h

    if box_w < 1e-4 or box_h < 1e-4:
        # print("not a valid box")
        return False

    # 计算中心点所在网格坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 计算中心点偏移量和宽高
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    # 计算边界框位置参数的损失权重 越大权重越低
    weight = 2.0 - (box_w / w)*(box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[]):
    # label_lists 是原始数据 gt_tensor是处理后的数据，计算出了中心位置，宽高和类别
    # nessary params
    batch_size = len(label_lists)
    w = input_size
    h = input_size
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])
    # 1+1+4+1，分别是置信度（1）、类别标签（1）、边界框（4）、边界框回归权重（1）

    # make labels
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[-1])
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                # 下标有效 填入true box的信息 如conf=1
                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x,
                              2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight

    gt_tensor = gt_tensor.reshape(batch_size, -1, 1+1+4+1)

    return torch.from_numpy(gt_tensor).float()

def compute_loss(pred_conf,pred_cls,pred_txtytwth,targets):

    # 损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    # 预测
    pred_conf = pred_conf[:, :, 0]           # [B, HW,]
    pred_cls = pred_cls.permute(0, 2, 1)     # [B, C, HW]
    pred_txty = pred_txtytwth[:, :, :2]      # [B, HW, 2]
    pred_twth = pred_txtytwth[:, :, 2:]      # [B, HW, 2]
    
    # 标签
    gt_obj = targets[:, :, 0]                  # [B, HW,]
    gt_cls = targets[:, :, 1].long()           # [B, HW,]
    gt_txty = targets[:, :, 2:4]               # [B, HW, 2]
    gt_twth = targets[:, :, 4:6]               # [B, HW, 2]
    gt_box_scale_weight = targets[:, :, 6]     # [B, HW,]

    batch_size = pred_conf.size(0)
    # conf loss 对象是所有的bbox(grid cell)
    conf_loss = conf_loss_function(pred_conf,gt_obj)
    conf_loss = conf_loss.sum() / batch_size

    # class loss 对象是所有的grid cell
    cls_loss = cls_loss_function(pred_cls,gt_cls) * gt_obj
    cls_loss = cls_loss.sum() / batch_size

    # bounding tx ty loss
    txty_loss = txty_loss_function(pred_txty,gt_txty).sum(-1) * gt_obj * gt_box_scale_weight
    txty_loss = txty_loss.sum() / batch_size

    # bounding tw th loss
    twth_loss = twth_loss_function(pred_twth,gt_twth).sum(-1) * gt_obj * gt_box_scale_weight
    twth_loss = twth_loss.sum() / batch_size
    bbox_loss = txty_loss + twth_loss

    # total loss
    total_loss = conf_loss + cls_loss + bbox_loss

    return conf_loss,cls_loss,bbox_loss,total_loss


if __name__ == "__main__":
    pass