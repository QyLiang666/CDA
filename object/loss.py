import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        #fill_diagonal_ :将mask的对角线元素设为0
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num

        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).long().cuda()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1).cuda()
        loss = self.criterion(logits, labels)
        loss /= N

        return loss,ne_loss

def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    sample_per_class = torch.tensor(sample_per_class)
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def MSL(input_):
    msl = torch.sum(input_**2, dim=1)
    return -1/2*msl

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def Negative_Entropy(softmax_out):
    negative_label = torch.zeros_like(softmax_out)
    negative_label[softmax_out < 0.05] = 1
    epsilon = 1e-5
    negative_entropy = - negative_label * torch.log(1 - softmax_out + epsilon)
    negative_entropy = torch.sum(negative_entropy, dim=1)
    return negative_entropy

def compute_Ldist(source_netF, source_netB, target_netF, target_netB, lambda1):
    Ldist = torch.tensor(0.0, requires_grad=True).cuda()

    # 遍历netF的每一层
    for source_module, target_module in zip(source_netF.named_modules(), target_netF.named_modules()):
        source_name, source_layer = source_module
        target_name, target_layer = target_module

        # 判断当前层是否为BatchNorm层
        if isinstance(source_layer, nn.BatchNorm2d) and isinstance(target_layer, nn.BatchNorm2d):
            # 获取源域模型和目标域模型的BN层的运行统计量
            source_mean = source_layer.running_mean.cuda()
            
            source_var = source_layer.running_var.cuda()
            target_mean = target_layer.running_mean.cuda()

            target_var = target_layer.running_var.cuda()
            # 计算均值差异和方差差异
            mean_diff = torch.norm(target_mean - source_mean, p=2)  # 使用L2范式计算差异
            var_diff = torch.norm(target_var - source_var, p=2)

            Ldist += (mean_diff + var_diff)

    # 遍历netB的每一层
    for source_module, target_module in zip(source_netB.named_modules(), target_netB.named_modules()):
        source_name, source_layer = source_module
        target_name, target_layer = target_module

        # 判断当前层是否为BatchNorm层
        if isinstance(source_layer, nn.BatchNorm1d) and isinstance(target_layer, nn.BatchNorm1d):
            # 获取源域模型和目标域模型的BN层的运行统计量
            # print(source_layer)
            # print(target_layer)
            # print(source_mean)
            # print(target_mean)
            source_mean = source_layer.running_mean.cuda()
            source_var = source_layer.running_var.cuda()
            target_mean = target_layer.running_mean.cuda()
            target_var = target_layer.running_var.cuda()

            # 计算均值差异和方差差异
            mean_diff = torch.norm(target_mean - source_mean, p=2)  # 使用L2范式计算差异
            var_diff = torch.norm(target_var - source_var, p=2)
            # print("mean {}".format(mean_diff))
            # print("var {}".format(var_diff))
            Ldist += (mean_diff + var_diff)
            
            


    # 添加权重正则化项
    weight_reg_loss = 0.0
    for source_param, target_param in zip(source_netF.parameters(), target_netF.parameters()):
        weight_reg_loss += torch.norm(target_param - source_param, p=1)
    for source_param, target_param in zip(source_netB.parameters(), target_netB.parameters()):
        weight_reg_loss += torch.norm(target_param - source_param, p=1)

    print("mean_var_diff {}".format(Ldist))
    print("weight_diff: {}".format(weight_reg_loss* lambda1))
    # 计算最终的Ldist
    Ldist += lambda1 * weight_reg_loss
    return Ldist

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss
