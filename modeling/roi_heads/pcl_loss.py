import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class PCLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pcl_probs, labels, cls_weights,
                gt_assignment, pc_labels, pc_probs, pc_count,
                img_cls_weights, im_labels):
        ctx.pcl_probs = pcl_probs
        ctx.labels = labels
        ctx.cls_weights = cls_weights
        ctx.gt_assignment = gt_assignment
        ctx.pc_labels = pc_labels
        ctx.pc_probs = pc_probs
        ctx.pc_count = pc_count
        ctx.img_cls_weights = img_cls_weights
        ctx.im_labels = im_labels
        
        batch_size, channels = pcl_probs.size()
        loss = 0
        ctx.mark_non_differentiable(labels, cls_weights,
                                    gt_assignment, pc_labels, pc_probs,
                                    pc_count, img_cls_weights, im_labels)
        for im_label in im_labels:
            if im_label == (channels - 1):
                labels_mask = (labels == im_label).nonzero()[:,0]
                loss -= (cls_weights[labels_mask] * pcl_probs.index_select(1, im_label).squeeze(-1)[labels_mask].log()).sum()
            else:
                labels_mask = (pc_labels == im_label).nonzero()[:,0]
                loss -= (img_cls_weights[labels_mask] * pc_probs[labels_mask].log()).sum()
        return loss / batch_size
    
    @staticmethod
    def backward(ctx, grad_output):
        pcl_probs = ctx.pcl_probs
        labels = ctx.labels
        cls_weights = ctx.cls_weights
        gt_assignment = ctx.gt_assignment
        pc_labels = ctx.pc_labels
        pc_probs = ctx.pc_probs
        pc_count = ctx.pc_count
        img_cls_weights = ctx.img_cls_weights
        im_labels = ctx.im_labels

        grad_input = grad_output.new(pcl_probs.size()).zero_()

        batch_size, channels = pcl_probs.size()

        for im_label in im_labels:
            labels_mask = (labels == im_label)
            if im_label == (channels - 1):
                grad_input[labels_mask, im_label] = -cls_weights[labels_mask]/pcl_probs[labels_mask, im_label]
            else:
                pc_index = gt_assignment[labels_mask]
                if (im_label != pc_labels[pc_index]).all():
                    print ("Labels Mismatch.")
                grad_input[labels_mask, im_label] = -img_cls_weights[pc_index] / (pc_count[pc_index] * pc_probs[pc_index])
        
        grad_input /= batch_size
        return grad_input, None, None, None, None, None, None, None, None