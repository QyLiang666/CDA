import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_PA, count_elements
from loss import ClusterLoss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix as cm
import random
import subprocess
import time
import wandb
import math
import copy
from PIL import ImageFilter
import matplotlib.pyplot as plt
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

def compute_adjustment(class_distribution, tro):
    adjustments = torch.log(class_distribution ** tro + 1e-6)
    return adjustments

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

class ClassDistributionUpdater:
    def __init__(self, args):
        self.global_class_distribution = None
        self.ema_weight = args.distribution_ema
        self.class_num = args.class_num
        self.uniform_init = args.uniform_init
    def update(self, pred_class_distribution):
        if self.global_class_distribution is None:
            if args.uniform_init:
                # 用均匀分布初始化
                self.global_class_distribution = (torch.ones(self.class_num) * (1.0 / self.class_num)).cuda().float()
            else:
                # 用第一轮训练开始前模型预测的概率分布初始化
                self.global_class_distribution = pred_class_distribution
        else:
            self.global_class_distribution =  self.global_class_distribution * self.ema_weight + pred_class_distribution * (1 - self.ema_weight)
        return self.global_class_distribution

class SelectRatioUpdater:

    def __init__(self, args):
        self.epoch_select_num = None
        self.train_data_num = args.train_data_num

    def update(self, batch_select_num):
        if self.epoch_select_num is None:
            self.epoch_select_num = batch_select_num
        else:
            self.epoch_select_num += batch_select_num 
    def select_ratio(self):
        a = self.epoch_select_num
        self.epoch_select_num = None
        return a / self.train_data_num

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, img):
        out1 = self.transform(img)
        out2 = self.transform2(img)
        out3 = self.transform2(img)
        return out1, out2, out3

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    
def image_simclraugment(resize_size=256, crop_size=224):
    return transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

def image_autoaugment(resize_size=256, crop_size=224):

    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.autoaugment.RandAugment(num_ops=2, magnitude=9),   
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        CutoutDefault(16)
    ])


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()     
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList(txt_tar, transform=image_train(), cfg=args, balance_sample=False)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test(), cfg=args, balance_sample=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    args.train_data_num = len(dsets["target"])
    return dset_loaders, txt_tar


def cal_acc(loader, netF, netB, netC, visda_flag=False, Test=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                # all_input = inputs.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                # all_input = torch.cat((all_input, inputs.float()), 0)
    _, predict = torch.max(all_output, 1)
    
    # if(Test==True):
    #     wandb_img_visual(all_input, predict, all_label)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracy *= 100
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    confusion_matrix = cm(all_label, torch.squeeze(predict).float())
    per_cls_acc_vec = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1) * 100
    per_cls_avg_acc = per_cls_acc_vec.mean()    # Per-class avg acc
    per_cls_acc_list = [str(np.round(i, 2)) for i in per_cls_acc_vec]
    acc_each_cls = ' '.join(per_cls_acc_list)

    if visda_flag:
        # For VisDA, return acc of each cls to be printed
        # overall acc, acc of each cls: str, per-class avg acc
        return accuracy, acc_each_cls, per_cls_avg_acc

    else:
        # For other datasets, need not return acc of each cls
        # overall acc, acc of each cls: str, mean-ent
        return accuracy, per_cls_avg_acc, mean_ent

def train_target(args):
    dset_loaders, txt_tar = data_load(args)
    dsets = dict()
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck, bias=True).cuda() 

    if args.abc :
        netC_abc = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck, bias=True).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    if args.abc :
        modelpath = args.output_dir_src + '/source_C.pt'
        netC_abc.load_state_dict(torch.load(modelpath))

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        v.requires_grad = False
    if args.abc :
        for k, v in netC_abc.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    #基于伪标签的训练正式开始
    max_iter = args.max_epoch * len(dset_loaders["target"])
    iter_per_epoch = len(dset_loaders["target"])
    print("Iter per epoch: {}".format(iter_per_epoch))
    interval_iter = max_iter // args.interval
    
    iter_num = 0

    if args.paral:
        netF = torch.nn.DataParallel(netF)
        netB = torch.nn.DataParallel(netB)
        netC = torch.nn.DataParallel(netC)

    netF.train()
    netB.train()
    netC.eval()
    if args.abc :
        netC_abc.train()

    if args.scd_lamb:
        scd_lamb_init = args.scd_lamb   # specify hyperparameter for secondary label correcion manually
    else:
        if args.dset[0:5] == "VISDA" :
            scd_lamb_init = 0.1
        elif args.dset[0:11] == "office-home":
            scd_lamb_init = 0.2
            if args.s == 3 and args.t == 2:
                scd_lamb_init *= 0.1
        elif args.dset[0:9] == "domainnet":
            scd_lamb_init = 0.02

    scd_lamb = scd_lamb_init

    class_distribution = ClassDistributionUpdater(args)
    consistency_select_ratio = SelectRatioUpdater(args)
    consistency_select_ratio_abc = SelectRatioUpdater(args)
    while iter_num < max_iter:
        k = 1.0
        k_s = 0.6   

        if iter_num % interval_iter == 0 and args.cls_par > 0:  # interval_itr = itr per epoch
            netF.eval()
            netB.eval()
            if args.abc:
                netC_abc.eval()
            label_prob_dict, samples_per_class, pseudo_label_acc = obtain_label(dset_loaders['test'], netF, netB, netC, args)   # dset_loaders['test'] 的shuffle为false，因此是original dataset的顺序  
            
            # 防止某类别出现样本数为0的情况
            samples_per_class = np.maximum(samples_per_class, 1)
            
            pred_class_distribution = samples_per_class / np.sum(samples_per_class)
            pred_class_distribution = torch.from_numpy(pred_class_distribution).cuda().float()
            global_class_distribution = class_distribution.update(pred_class_distribution)
            # print("pred_class_distribution: {}".format(pred_class_distribution))

            mem_label, pseudo_lb_prob = label_prob_dict['primary_lb'], label_prob_dict['primary_lb_prob']
            mem_label, pseudo_lb_prob = torch.from_numpy(mem_label).cuda(), torch.from_numpy(pseudo_lb_prob).cuda()
            if args.scd_label:
                second_label, second_prob = label_prob_dict['secondary_lb'], label_prob_dict['secondary_lb_prob']
                second_label, second_prob = torch.from_numpy(second_label).cuda(), torch.from_numpy(second_prob).cuda()
            if args.third_label:
                third_label, third_prob = label_prob_dict['third_lb'], label_prob_dict['third_lb_prob']
                third_label, third_prob = torch.from_numpy(third_label).cuda(), torch.from_numpy(third_prob).cuda()
            if args.fourth_label:
                fourth_label, fourth_prob = label_prob_dict['fourth_lb'], label_prob_dict['fourth_lb_prob']
                fourth_label, fourth_prob = torch.from_numpy(fourth_label).cuda(), torch.from_numpy(fourth_prob).cuda()
            if args.topk_ent:
                all_entropy = label_prob_dict['entropy']
                all_entropy = torch.from_numpy(all_entropy)

            if args.dset[0:5] == "VISDA" :
                if iter_num // iter_per_epoch < 1:
                    k = 0.6
                elif iter_num // iter_per_epoch < 2:
                    k = 0.7
                elif iter_num // iter_per_epoch < 3:
                    k = 0.8
                elif iter_num // iter_per_epoch < 4:
                    k = 0.9
                else:
                    k = 1.0

                if iter_num // iter_per_epoch >= 8:
                    scd_lamb *= 0.1

            elif args.dset[0:11] == "office-home" or args.dset[0:9] == "domainnet":
                if iter_num // iter_per_epoch < 2:
                    k = 0.2
                elif iter_num // iter_per_epoch < 4:
                    k = 0.4
                elif iter_num // iter_per_epoch < 8:
                    k = 0.6
                elif iter_num // iter_per_epoch < 12:
                    k = 0.8

            if args.no_threshold_k:
                k = 1.0

            if args.topk:
                dsets["target"] = ImageList_PA(txt_tar, mem_label, pseudo_lb_prob, k_low=k, k_up=None,
                                                transform=TransformTwice(image_train(), image_autoaugment()))
                dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.worker, drop_last=False)
            

            if args.topk_ent:
                dsets["target"] = ImageList_PA(txt_tar, mem_label, -1.0 * all_entropy, k_low=k, k_up=None,
                                                transform=TransformTwice(image_train(), image_autoaugment()))
                dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.worker, drop_last=False)

            if args.scd_label:
                # 2nd label threshold: prob top 60%
                dsets["target_scd"] = ImageList_PA(txt_tar, second_label, second_prob, k_low=k_s, k_up=None,
                                                    transform=image_train())
                dset_loaders["target_scd"] = DataLoader(dsets["target_scd"], batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.worker, drop_last=False)
            netF.train()
            netB.train()
            if args.abc:
                netC_abc.train()
        if args.topk_ent:
            try:
                (inputs_weak, inputs_strong1, inputs_strong2), _, tar_idx = next(iter_train)
            except:
                iter_train = iter(dset_loaders["target"])
                (inputs_weak, inputs_strong1, inputs_strong2), _, tar_idx = next(iter_train)  # tar_idx: chosen indices in current itr
        else:
            try:
                inputs_weak, tar_idx = next(iter_train)
            except:
                iter_train = iter(dset_loaders["target"])
                inputs_weak, tar_idx = next(iter_train)

        if inputs_weak.size(0) == 1:
            continue

        if args.scd_label:
            try:
                inputs_scd, _, tar_idx_scd = next(iter_scd)
            except:
                iter_scd = iter(dset_loaders["target_scd"])
                inputs_scd, _, tar_idx_scd = next(iter_scd)

            if inputs_scd.size(0) == 1:
                continue   

        iter_num += 1
        # print(iter_num)
        inputs_weak = inputs_weak.cuda()
        features_weak = netB(netF(inputs_weak))
        logits_weak = netC(features_weak)
        softmax_weak = nn.Softmax(dim=1)(logits_weak)
        pred_value_weak, pred_weak = torch.max(softmax_weak, dim=1)

        select_num = pred_value_weak.ge(args.consistency_threshold).sum() # 计算伪标签中1的个数，即大于阈值的个数
        consistency_select_ratio.update(select_num) 

        if args.autoaugment or args.abc:
            inputs_strong1 = inputs_strong1.cuda()
            inputs_strong2 = inputs_strong2.cuda()

            inputs_strong = interleave(torch.cat((inputs_strong1, inputs_strong2)), 2)
            features_strong = netB(netF(inputs_strong))
            logits_strong = netC(features_strong)

            logits_strong = de_interleave(logits_strong, 2)
            logits_strong1, logits_strong2 = logits_strong.chunk(2)
            del logits_strong

            softmax_strong1 = nn.Softmax(dim=1)(logits_strong1)
            softmax_strong2 = nn.Softmax(dim=1)(logits_strong2)

            if args.abc:
                # logits_weak_abc = netC_abc(features_weak)
                # softmax_weak_abc = nn.Softmax(dim=1)(logits_weak_abc)
                # pred_value_abc, predict_weak_abc = torch.max(softmax_weak_abc, 1)

                logits_strong_abc = netC_abc(features_strong)
                logits_strong_abc = de_interleave(logits_strong_abc, 2)
                logits_strong1_abc, logits_strong2_abc = logits_strong_abc.chunk(2)
                del logits_strong_abc

                softmax_strong1_abc = nn.Softmax(dim=1)(logits_strong1_abc)
                softmax_strong2_abc = nn.Softmax(dim=1)(logits_strong2_abc)
        unknown_weight = label_prob_dict['unknown_weight']
        unknown_weight = unknown_weight[tar_idx]                                          
        unknown_weight = torch.from_numpy(unknown_weight).float().cuda()

        # 这是原始图片通过kmeans得到的伪标签，每个epoch之前都要用kmeans生成一遍伪标签
        pred = mem_label[tar_idx]

        if args.scd_label:
            inputs_scd = inputs_scd.cuda()
            if inputs_scd.ndim == 3:
                inputs_scd = inputs_scd.unsqueeze(0)

            features_scd = netB(netF(inputs_scd))
            logits_scd = netC(features_scd)

            first_prob_of_scd = pseudo_lb_prob[tar_idx_scd]
            scd_prob = second_prob[tar_idx_scd]
            if not args.no_mask:
                mask = (scd_prob / first_prob_of_scd.float()).clamp(max=1.0)
            else:
                mask = torch.ones_like(scd_prob).cuda()

        if args.intra_dense or args.inter_sep:
            intra_dist = torch.zeros(1).cuda()  
            inter_dist = torch.zeros(1).cuda()
            same_first = True
            diff_first = True
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            for i in range(pred.size(0)):
                for j in range(i, pred.size(0)):
                    # dist = torch.norm(features_weak[i] - features_weak[j])
                    dist = 0.5 * (1 - cos(features_weak[i].unsqueeze(0), features_weak[j].unsqueeze(0)))   
                    if pred[i].item() == pred[j].item():
                        if same_first:
                            intra_dist = dist.unsqueeze(0)
                            same_first = False
                        else:
                            intra_dist = torch.cat((intra_dist, dist.unsqueeze(0)))

                    else:
                        if diff_first:
                            inter_dist = dist.unsqueeze(0)
                            diff_first = False
                        else:
                            inter_dist = torch.cat((inter_dist, dist.unsqueeze(0)))

            intra_dist = torch.mean(intra_dist)
            inter_dist = torch.mean(inter_dist)

        if args.cls_par > 0:
            classifier_loss_none = nn.CrossEntropyLoss(reduction='none')(logits_weak, pred).cuda()  # self-train by pseudo label

            if not args.unknown_weight: 
                classifier_loss = torch.mean(classifier_loss_none)
                classifier_loss = classifier_loss * args.cls_par

            # unknown_weight
            else:
                classifier_loss = classifier_loss_none * unknown_weight
                classifier_loss = torch.mean(classifier_loss) * args.unknown_weight_par

            if args.scd_label:
                pred_scd = second_label[tar_idx_scd]
                classifier_loss_scd = nn.CrossEntropyLoss(reduction='none')(logits_scd , pred_scd).cuda()  # self-train by pseudo label
                classifier_loss_scd = torch.mean(mask * classifier_loss_scd)
                classifier_loss_scd *= args.cls_par
                classifier_loss += classifier_loss_scd * scd_lamb

        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.negative_learning:
            negative_loss = torch.mean(loss.Negative_Entropy(softmax_weak))
            classifier_loss += negative_loss

        if iter_num < interval_iter :
            classifier_loss *= 0           # 默认第一轮标签不够准确，所以第一轮时classifier_loss=0，只用IM去更新。
        
        if args.autoaugment: 
            select_mask_aug = pred_value_weak.ge(args.consistency_threshold)
            consistency_loss = -torch.mean(select_mask_aug * torch.sum(torch.log(softmax_strong1) * softmax_weak.cuda().detach(), dim=1))
            consistency_loss += -torch.mean(select_mask_aug * torch.sum(torch.log(softmax_strong2) * softmax_weak.cuda().detach(), dim=1))
            if iter_num < interval_iter :
                consistency_loss *= 0
            classifier_loss += consistency_loss * args.autoaugment_par

        if args.abc:
            # classifier_loss_abc = select_mask * loss.balanced_softmax_loss(predict_weak_abc, logits_weak_abc, global_class_distribution, reduction = "none")
            adjustment = compute_adjustment(global_class_distribution, tro=args.tao)
            softmax_weak_abc = torch.softmax((logits_weak.detach() - adjustment), dim=-1)
            pred_value_abc, predict_weak_abc = torch.max(softmax_weak_abc, 1)
            select_mask_abc = pred_value_abc.ge(args.consistency_threshold) | pred_value_weak.ge(args.consistency_threshold)
            select_num_abc = select_mask_abc.sum()      # 计算select_mask中1的个数，即大于阈值的个数
            consistency_select_ratio_abc.update(select_num_abc) 

            consistency_loss_abc = -torch.mean(select_mask_abc * torch.sum(torch.log(softmax_strong1_abc) * softmax_weak_abc.cuda().detach(), dim=1))
            consistency_loss_abc += -torch.mean(select_mask_abc * torch.sum(torch.log(softmax_strong2_abc) * softmax_weak_abc.cuda().detach(), dim=1))
            if iter_num < args.abc_begin_epoch * interval_iter :
                consistency_loss_abc *= 0
            classifier_loss += consistency_loss_abc * args.abc_par    

        if args.ent:
            entropy_loss = torch.mean(loss.Entropy(softmax_weak))  # Minimize local entropy
            if args.msl != 0.0:
                entropy_loss = torch.mean(loss.MSL(softmax_weak)) * args.msl

            if args.scd_label:
                softmax_out_scd = nn.Softmax(dim=1)(logits_scd)
                entropy_loss_scd = torch.mean(mask * loss.Entropy(softmax_out_scd))  # Minimize local entropy

            if args.gent:
                msoftmax = softmax_weak.mean(dim=0)
                if args.gent_decay : # cos decay
                    gent_decay_par = math.cos((math.acos(args.gent_decay_weight))/args.max_epoch*iter_num/iter_per_epoch)
                    gentropy_loss = -torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))*gent_decay_par
                    entropy_loss += gentropy_loss  # Maximize global entropy
                elif args.gent_sigmoid_decay:
                    # y = (2 / (1 + np.exp(k * (x - 1))))-1
                    gent_smooth_par = (1/(1+math.exp(10*(iter_num/max_iter-0.5))))
                    # gent_smooth_par = 1/(1+math.exp(-5*(iter_num/max_iter-0.5)))
                    gentropy_loss = -torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))*gent_smooth_par
                    entropy_loss += gentropy_loss  # Maximize global entropy
                elif args.gent_line_decay:
                    # y = 1-x
                    gent_smooth_par = 1-iter_num/max_iter
                    gentropy_loss = -torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))*gent_smooth_par
                    entropy_loss += gentropy_loss  # Maximize global entropy
                else:    
                    gentropy_loss = -torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss += gentropy_loss  # Maximize global entropy

                if args.scd_label:
                    msoftmax_scd = softmax_out_scd.mean(dim=0)
                    gentropy_loss_scd = -torch.sum(-msoftmax_scd * torch.log(msoftmax_scd + args.epsilon))
                    entropy_loss_scd += gentropy_loss_scd  # Maximize global entropy

            im_loss = entropy_loss * args.ent_par
            
            if args.scd_label:
                im_loss += entropy_loss_scd * args.ent_par * scd_lamb
            classifier_loss += im_loss

        if args.intra_dense:
            classifier_loss += args.lamb_intra * intra_dist.squeeze()   
        if args.inter_sep:
            classifier_loss += args.lamb_inter * inter_dist.squeeze()

        if args.unfreeze:
            if iter_num // iter_per_epoch == 10 :
                args.unfreeze = False
                for k, v in netC.named_parameters():
                    if args.lr_decay3 > 0:
                        v.requires_grad = True
                        param_group_new = []
                        param_group_new += [{'params': v, 'lr': args.lr * args.lr_decay3}]
                for group in param_group_new:
                    group['lr0'] = group['lr']
                    optimizer.add_param_group(group)

        optimizer.zero_grad()
        # print("classifier{}".format(classifier_loss))
        classifier_loss.backward()
        optimizer.step()
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
           
            # For all Dataset, print the acc of each cls
            acc_t_te, acc_list, acc_cls_avg = cal_acc(dset_loaders['test'], netF, netB, netC, visda_flag=True, Test=True)   
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Cls Avg Acc = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                        acc_t_te, acc_cls_avg) + '\n' + acc_list
            log_to_wandb = {'iter_num':iter_num, 'total_loss':classifier_loss.item(), 'Accuracy':acc_t_te, 'Cls_Avg_Acc':acc_cls_avg}
            if args.abc:
                acc_t_te_abc, acc_list_abc, acc_cls_avg_abc = cal_acc(dset_loaders['test'], netF, netB, netC_abc, visda_flag=True, Test=True)    # flag for VisDA -> need cls avg acc.
                log_str_abc = 'Task: {}, Iter:{}/{}; Accuracy Of ABC = {:.2f}%, Cls Avg Acc Of ABC= {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_t_te_abc, acc_cls_avg_abc) + '\n' + acc_list_abc
                log_to_wandb ['Accuracy_abc'] = acc_t_te_abc
                log_to_wandb ['Cls_Avg_Acc_abc'] = acc_cls_avg_abc

            # 统计每个loss的值，wandb分析
            if 'im_loss' in locals():
                log_to_wandb['im_loss'] = im_loss.item()
            if 'gentropy_loss' in locals():
                log_to_wandb['gentropy_loss'] = gentropy_loss.item()
            if 'consistency_loss' in locals():
                log_to_wandb['consistency_loss'] = consistency_loss.item()
            if 'negative_loss' in locals():
                log_to_wandb['negative_loss'] = negative_loss.item()
            if 'select_num' in locals():
                log_to_wandb['consistency_data_select_ratio'] = consistency_select_ratio.select_ratio()
            if 'select_num_abc' in locals():
                log_to_wandb['consistency_data_select_ratio_abc'] = consistency_select_ratio_abc.select_ratio()
            if 'pseudo_label_acc' in locals():
                log_to_wandb['pseudo_label_acc'] = pseudo_label_acc
            run.log(log_to_wandb)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            if args.abc:
                args.out_file.write(log_str_abc + '\n')
                args.out_file.flush()
                print(log_str_abc + '\n')
            netF.train()
            netB.train()

    if args.issave:
        if args.paral:
            torch.save(netF.module.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
            torch.save(netB.module.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
            torch.save(netC.module.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        else:        
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)  # output logits
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)  
    all_output = nn.Softmax(dim=1)(all_output)  # pred prob
    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) 
    _, predict = torch.max(all_output, 1)   
    all_entropy = torch.sum(-1.0 * all_output.float() * torch.log(all_output.float() + args.epsilon), dim=1).numpy()  
    unknown_weight = 1 - all_entropy / np.log(args.class_num)  
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])  

    if args.distance == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()             

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)                           
    aff = all_output.float().cpu().numpy()             
    initc = aff.transpose().dot(all_fea)               
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  

    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)  
    labelset = labelset[0]
    # print(labelset)
                                                       
    dd = cdist(all_fea, initc[labelset], args.distance) 
    # cosine_similarity = 1 - dd 
    pred_label = dd.argmin(axis=1)                    
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])    
        dd = cdist(all_fea, initc[labelset], args.distance)  
        pred_label = dd.argmin(axis=1)                       
        pred_label = labelset[pred_label]                    
    samples_per_class = count_elements(pred_label, args.class_num)

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)  # acc before & after clustering

    pseudo_lb_prob = np.zeros(pred_label.shape)
    for i in range(pred_label.shape[0]):
        pseudo_lb_prob[i] = all_output[i][pred_label[i]]

    label_prob_dict = dict()
    label_prob_dict['primary_lb'] = pred_label.astype('int')
    label_prob_dict['primary_lb_prob'] = pseudo_lb_prob
    label_prob_dict['entropy'] = all_entropy
    label_prob_dict['unknown_weight'] = unknown_weight
    label_prob_dict['prototype'] = initc

    ## Secondary labels
    if args.scd_label:
        second_lb = torch.zeros(predict.size()).numpy()
        second_prob = torch.zeros(predict.size()).numpy()

        for i in range(second_lb.shape[0]):
            idx = np.argsort(all_output[i].numpy())[-2]
            second_lb[i] = idx
            second_prob[i] = all_output[i][idx]

        label_prob_dict['secondary_lb'] = second_lb.astype('int')
        label_prob_dict['secondary_lb_prob'] = second_prob

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return label_prob_dict, samples_per_class, acc

def wait_for_GPU_avaliable(gpu_id):
    isVisited = False
    while True:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,nounits,noheader']).decode()
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        
        available_memory = 0

        for gpu_id in args.gpu_id.split(","):
            available_memory += gpu_memory_map[int(gpu_id)]

        # wait unless GPU memory is more than 10000
        if available_memory < 10000:
            if not isVisited:
                print("GPU full, wait...........")
                isVisited = True
            time.sleep(120)
            continue
        else:
            print("Empty GPU! Start process!")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15, help="为了每个epoch都测试一轮(都重新聚类一次),将interval设置为max_epoch")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['office-home-RSUT', 'domainnet', 'VISDA-RSUT', 'VISDA-RSUT-50', 'VISDA-RSUT-10',
                                 'VISDA-Beta', 'VISDA-Tweak', 'VISDA-Knockout', 'VISDA-C', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='../result/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--topk', default=False, action='store_true')
    parser.add_argument('--topk_ent', default=False, action='store_true')
    parser.add_argument('--scd_label', default=False, action='store_true')
    parser.add_argument('--scd_lamb', type=float, default=None)
    parser.add_argument('--third_label', default=False, action='store_true')
    parser.add_argument('--fourth_label', default=False, action='store_true')
    parser.add_argument('--intra_dense', default=False, action='store_true')
    parser.add_argument('--inter_sep', default=False, action='store_true')
    parser.add_argument('--no_mask', default=False, action='store_true')
    parser.add_argument('--lamb_intra', type=float, default=1.0)
    parser.add_argument('--lamb_inter', type=float, default=-0.1)
    parser.add_argument('--paral', default=False, action='store_true')

    parser.add_argument('--unknown_weight', default=False, action='store_true')
    parser.add_argument('--gent_decay', default=False, action='store_true')
    parser.add_argument('--gent_decay_weight', type=float, default=0.8)    
    parser.add_argument('--msl', type=float, default=0.0)
    parser.add_argument('--negative_learning', default=False, action='store_true')
    parser.add_argument('--no_threshold_k', default=False, action='store_true')
    parser.add_argument('--autoaugment', default=False, action='store_true')
    parser.add_argument('--unfreeze', default=False, action='store_true')
    parser.add_argument('--gent_sigmoid_decay', default=False, action='store_true')
    parser.add_argument('--gent_line_decay', default=False, action='store_true')
    parser.add_argument('--abc', default=False, action='store_true')
    parser.add_argument('--distribution_ema', type=float, default=0.6)
    parser.add_argument('--consistency_threshold', type=float, default=0.8)
    parser.add_argument('--tao', type=float, default=1)
    parser.add_argument('--select_ratio_ema', type=float, default=0.9)
    parser.add_argument('--uniform_init', default=False, action='store_true')
    parser.add_argument('--abc_begin_epoch', type=int, default=1, help="from (abc_begin_epoch + 1) to max_epoch, use abc")
    parser.add_argument('--unknown_weight_par', type=float, default=1)
    parser.add_argument('--autoaugment_par', type=float, default=1)
    parser.add_argument('--abc_par', type=float, default=1)
    parser.add_argument('--wandb_new_project_name', type=str, default="new_project")
    args = parser.parse_args()
    args.interval = args.max_epoch


    if args.dset == 'office-home-RSUT' or args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']  # only 1,2,3 are available
        args.class_num = 65

    if args.dset == 'domainnet':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 40

    if args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10' \
            or args.dset == 'VISDA-Tweak' or args.dset == 'VISDA-Knockout' or args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12

    if args.dset == 'VISDA-Beta':
        names = ['train_b1_a1', 'validation_b2.0_a1.2', 'validation_b2.0_a2.0', 'validation_b2.0_a2.7']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    wait_for_GPU_avaliable(args.gpu_id)

    for i in range(len(names)):
        if i == args.s:
            continue
        if args.dset == 'office-home-RSUT' and names[i] == 'Art':
            continue
        args.t = i

        folder = '../data/'
        if args.dset == 'office-home-RSUT':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
        elif args.dset == 'domainnet':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_train_mini.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'
        elif args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
        else:
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
        wandb_name = names[args.s][0].upper() + "->" + names[args.t][0].upper() 
        print(wandb_name)
        project_name = "Imbalanced-SFDA" + "_" + args.dset + args.wandb_new_project_name
        with wandb.init(project = project_name, config = args.__dict__, name = wandb_name, save_code=True)  as run :

            args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())

            if args.dset != 'VISDA-Beta':
                args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() +
                                        names[args.t][0].upper())
                args.name = names[args.s][0].upper() + names[args.t][0].upper()
            else:
                args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() +
                                        names[args.t][0].upper() + names[args.t][-4:])
                args.name = names[args.s][0].upper() + names[args.t][0].upper() + names[args.t][-4:]

            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)

            args.savename = 'par_' + str(args.cls_par)
            args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            args.out_file.flush()

            train_target(args)