__author__ = 'YaelSegal & TzeviyaFuchs'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YOLOLoss:
    def __init__(self, noobject_conf=0.5, obj_conf=1, coordinate=10, class_conf=1, loss_type="mse"):
        self.noobject_conf = noobject_conf
        self.obj_conf = obj_conf
        self.coordinate = coordinate
        self.class_conf = class_conf
        self.loss_type = loss_type

    def loss(self, y_pred, y_true, config_dict, use_cuda=False):
        """

        :param y_pred: (Batch, C * (3*B + K)
        :param y_true: (C *( B*3+ K +1))
        :param config_dict: {"C":c, "B":b, "K":k}
        :param use_cuda: run on GPU or CPU
        :return: loss value
        """

        C = config_dict["C"]
        B = config_dict["B"]
        K = config_dict["K"]

        noobject_conf = self.noobject_conf
        obj_conf = self.obj_conf
        coordinate = self.coordinate
        class_conf = self.class_conf

        # targets
        target_coords = y_true[:, :, :3 * B].contiguous().view(-1, C, B, 3)  # get all the x,w values for all the boxes
        target_xs = target_coords[:, :, :, 0].view(-1, C, B, 1)
        target_xs_no_norm = target_coords[:, :, :, 0].view(-1, C, B, 1) / float(C)
        target_ws = torch.pow(target_coords[:, :, :, 1].view(-1, C, B, 1), 2)  # assuming the prediction is for sqrt(w)
        target_conf = target_coords[:, :, :, 2].view(-1, C, B, 1)
        target_start = target_xs_no_norm - (target_ws * 0.5)
        target_end = target_xs_no_norm + (target_ws * 0.5)
        target_class_prob = y_true[:, :, 3 * B:-1].contiguous().view(-1, C, K, 1)

        # pred
        pred_coords = y_pred[:, :, :3 * B].contiguous().view(-1, C, B, 3)  # get all the x,w values for all the boxes
        pred_xs = pred_coords[:, :, :, 0].view(-1, C, B, 1)
        pred_xs_no_norm = pred_coords[:, :, :, 0].view(-1, C, B, 1) / float(C)
        pred_ws = torch.pow(pred_coords[:, :, :, 1].view(-1, C, B, 1), 2)  # assuming the prediction is for sqrt(w)
        pred_conf = pred_coords[:, :, :, 2].view(-1, C, B, 1)
        pred_start = pred_xs_no_norm - (pred_ws * 0.5)
        pred_end = pred_xs_no_norm + (pred_ws * 0.5)
        pred_class_prob = y_pred[:, :, 3 * B:].contiguous().view(-1, C, K, 1)
        # Calculate the intersection areas
        intersect_start = torch.max(pred_start, target_start)
        intersect_end = torch.min(pred_end, target_end)
        intersect_w = intersect_end - intersect_start

        # Calculate the best IOU, set 0.0 confidence for worse boxes
        iou = intersect_w / (pred_ws + target_ws - intersect_w)
        iou_max_value, iou_max_indices = torch.max(iou, 2)
        best_box = torch.eq(iou, iou_max_value.unsqueeze(2))
        one_confs_per_cell = best_box.float() * target_conf

        real_exist = (y_true[:, :, -1]).unsqueeze(2)  # the last place in y_true determines if the object exists
        obj_exists_classes = real_exist.repeat((1, 1, K)).view(-1, C, K, 1)
        obj_exists = one_confs_per_cell
        noobj_exists = torch.zeros([obj_exists.size(0), C, B, 1], dtype=torch.float32).cuda() if use_cuda else \
            torch.zeros([obj_exists.size(0), C, B, 1], dtype=torch.float32)
        noobj_exists = torch.eq(one_confs_per_cell, noobj_exists).float()

        if self.loss_type == "abs":
            first_part = torch.sum(make_flatt(coordinate * obj_exists * torch.abs((pred_xs - target_xs))), 1)
            second_part = torch.sum(make_flatt(5 * coordinate * obj_exists * torch.abs((pred_ws - target_ws))), 1)
        else:
            first_part = torch.sum(make_flatt(coordinate * obj_exists * torch.pow((pred_xs - target_xs), 2)), 1)
            second_part = torch.sum(make_flatt(coordinate * obj_exists * torch.pow((pred_ws - target_ws), 2)), 1)

        third_part = torch.sum(make_flatt(obj_conf * obj_exists * torch.pow((pred_conf - one_confs_per_cell), 2)), 1)
        fourth_part = torch.sum(
            make_flatt(noobject_conf * noobj_exists * torch.pow((pred_conf - one_confs_per_cell), 2)), 1)
        fifth_part = torch.sum(
            make_flatt(class_conf * obj_exists_classes * torch.pow((target_class_prob - pred_class_prob), 2)), 1)

        total_loss = first_part + second_part + third_part + fourth_part + fifth_part

        return torch.mean(total_loss), torch.mean(first_part), torch.mean(second_part), torch.mean(third_part), \
               torch.mean(fourth_part), torch.mean(fifth_part)


def make_flatt(table):
    return table.view(table.size(0), -1)
