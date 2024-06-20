import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, count_elements
import random
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix as cm
from sklearn.cluster import KMeans
import subprocess
import time
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from wandb_img_visualization import visualize_feature, visualize_confusion_matrix, visualize_pred_class_distribution

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

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, img):
        out1 = self.transform(img)
        out2 = self.transform2(img)
        return out1, out2

def image_autoaugment(resize_size=256, crop_size=224):

    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
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


def getClassDict(total_list):
    cls_dict = dict()
    for l in total_list:
        if not int(l.split(' ')[1]) in cls_dict:
            cls_dict[int(l.split(' ')[1])] = []
        cls_dict[int(l.split(' ')[1])].append(l)

    return cls_dict

def getSampleDict(total_cls_dict, per_cls_percentage):
    sample_cls_dict = dict()
    for k in total_cls_dict.keys():
        if not k in sample_cls_dict:
            sample_cls_dict[k] = []
        this_cls_num = len(total_cls_dict[k])
        val_num = max(int(this_cls_num * per_cls_percentage), 1)
        sample_cls_dict[k] = random.sample(total_cls_dict[k], val_num)

    return sample_cls_dict


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if args.trte == "val":  # 训练集和测试集按9:1随机划分
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])

    elif args.trte == "stratified": # 按照原始数据集中的各类别的比例划分训练集和测试集，相当于从每个类别中抽取10%的样本作为测试集
        cls_dict = getClassDict(txt_src)
        val_sample_cls_dict = getSampleDict(cls_dict, 0.1)
        te_txt = []
        for k in val_sample_cls_dict.keys():
            te_txt.extend(val_sample_cls_dict[k])
        tr_txt = list(set(txt_src) - set(te_txt))

    else:
        dsize = len(txt_src)
        tr_size = int(0.8 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    # training set
    if args.source_balanced:
        # balanced sampler of source train
        dsets["source_tr"] = ImageList(tr_txt, transform=TransformTwice(image_train(), image_autoaugment()), cfg=args, balance_sample=True)
    else:
        dsets["source_tr"] = ImageList(tr_txt, transform=TransformTwice(image_train(), image_autoaugment()), cfg=args, balance_sample=False)
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    # validation set
    dsets["source_te"] = ImageList(te_txt, transform=image_test(), cfg=args, balance_sample=False)
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    # test set
    dsets["test"] = ImageList(txt_test, transform=image_test(), cfg=args, balance_sample=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders, dsets


def cal_acc(loader, netF=None, netB=None, netC=None, per_class_flag=False, visda_flag=False):
    """Calculate model accuracy on validation set or testing set
    :param loader: dataloader
    :param netF: feature extractor network
    :param netB: bottleneck network
    :param netC: classifier network
    :param per_class_flag: if True: calculatge per-class average accuracy
    :param visda_flag: if True: return acc of each class, else: no need to return acc of each class
    :return: overall acc, per-class average acc, str: acc of each class, mean entropy
    """
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
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
    visualize_feature(all_fea, all_label, args) 
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    
    #模型在测试集上预测的各类别样本数
    samples_per_class = count_elements(predict, args.class_num)
    # 对预测出的samples_per_class求概率分布
    pred_class_distribution = samples_per_class / np.sum(samples_per_class)
    pred_class_distribution = torch.from_numpy(pred_class_distribution).cuda().float()
    # print("pred_class_distribution: {}".format(pred_class_distribution))
    
    visualize_pred_class_distribution(pred_class_distribution, "pred_class_distribution",args)    
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracy *= 100  # overall accuracy
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()  # average entropy of classification results

    confusion_matrix = cm(all_label, torch.squeeze(predict).float())
    visualize_confusion_matrix(confusion_matrix, args)
    if per_class_flag:
        per_cls_acc_vec = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1) * 100
        per_cls_avg_acc = per_cls_acc_vec.mean()  # Per-class avg acc
        per_cls_acc_list = [str(np.round(i, 2)) for i in per_cls_acc_vec]
        acc_each_cls = ' '.join(per_cls_acc_list)   # str: acc of each class

    if visda_flag:
        # For VisDA, return acc of each cls to be printed
        # overall acc, acc of each cls: str, per-class avg acc
        return accuracy, acc_each_cls, per_cls_avg_acc

    elif per_class_flag:
        # For Office-Home and DomainNet, no need to return acc of each class
        # overall acc, per-class avg acc, average entropy
        return accuracy, per_cls_avg_acc, mean_ent

    else:
        # overall acc, mean-ent
        return accuracy, mean_ent


def train_source(args):
    dset_loaders, dsets = data_load(args)
    ## set base network
    if args.net[0:3] == 'res' or args.net[0:3] == 'vgg':
        if args.net[0:3] == 'res':
            netF = network.ResBase(res_name=args.net).cuda()
        else:
            netF = network.VGGBase(vgg_name=args.net).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck).cuda()   # classifier: bn
         # layer: wn
        if args.bias_0:   
            netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck, bias=False).cuda()
        else:
            netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck, bias=True).cuda()

        if args.abc:
            netC_abc = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck, bias=True).cuda()

        if args.resume:
            args.modelpath = args.output_dir_src + '/source_F.pt'
            netF.load_state_dict(torch.load(args.modelpath))
            args.modelpath = args.output_dir_src + '/source_B.pt'
            netB.load_state_dict(torch.load(args.modelpath))
            args.modelpath = args.output_dir_src + '/source_C.pt'
            netC.load_state_dict(torch.load(args.modelpath))

        param_group = []
        learning_rate = args.lr
        for k, v in netF.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate * 0.1}]
        for k, v in netB.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate * args.netB_lr}]
        for k, v in netC.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        if args.abc:
            for k, v in netC_abc.named_parameters():
                param_group += [{'params': v, 'lr': learning_rate}]
        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

    if args.paral:
        netF = torch.nn.DataParallel(netF)
        netB = torch.nn.DataParallel(netB)
        netC = torch.nn.DataParallel(netC)

    acc_init = 0.
    #打印一个epoch有多少iter 
    iter_per_epoch = len(dset_loaders["source_tr"])
    print("Iter per epoch: {}".format(iter_per_epoch))
    max_iter = args.max_epoch * iter_per_epoch
    print_loss_interval = 25
    interval_iter = iter_per_epoch  #每个epoch后都要test一次
    iter_num = 0

    if args.net[0:3] == 'res' or args.net[0:3] == 'vgg':
        netF.train()
        netB.train()
        netC.train()
        if args.abc:
            netC_abc.train()

    while iter_num < max_iter:
        try:
            (inputs_source, inputs_source_autoaugment), labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            (inputs_source, inputs_source_autoaugment), labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        features_source = netB(netF(inputs_source))
        outputs_source = netC(features_source)

        if args.abc:
            outputs_source_abc = netC_abc(features_source)

        # 不归"一"化，归一化后，算交叉熵与分类器层权重算点积时再乘s（s=20）！
        if args.normal_to_genhao20:
            outputs_source = outputs_source * 20

        if args.balanced_softmax:
            samples_per_class = dsets["source_tr"].count_samples_per_class()         # samples_pre_class是一个列表装着每个类别的样本数

            classifier_loss = loss.balanced_softmax_loss(labels_source, outputs_source, samples_per_class, reduction = "mean")
        elif args.abc:
            samples_per_class = dsets["source_tr"].count_samples_per_class()         # samples_pre_class是一个列表装着每个类别的样本数

            classifier_loss_abc = loss.balanced_softmax_loss(labels_source, outputs_source_abc, samples_per_class, reduction = "mean")

            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                                   labels_source)
            classifier_loss = classifier_loss + classifier_loss_abc
        else:

            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        if args.autoaugment:

            inputs_source_autoaugment = inputs_source_autoaugment.cuda()
            features_source_autoaugment = netB(netF(inputs_source_autoaugment))
            outputs_source_autoaugment = netC(features_source_autoaugment)

            softmax_out_autoaugment = nn.Softmax(dim=1)(outputs_source_autoaugment)

            consistency_loss = nn.CrossEntropyLoss(reduction='mean')(softmax_out_autoaugment, labels_source)    # 这里是fixmatch思想，用强数据增强的结果去和伪硬标签做交叉熵
            classifier_loss += consistency_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % print_loss_interval == 0:
            print("Iter:{:>4d}/{} | Classification loss on Source: {:.2f}".format(iter_num, max_iter,
                                                                              classifier_loss.item()))
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.abc:
                netC = netC_abc
                netC.eval()
            else:
                netC.eval()
            if args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10' or args.dset == 'VISDA-C':
                # The small classes in VisDA-C (RSUT) still have relatively many samples.
                # Safe to use per-class average accuracy.
                acc_s_te, acc_list, acc_cls_avg_te= cal_acc(dset_loaders['source_te'], netF, netB, netC,
                                                            per_class_flag=True, visda_flag=True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Cls Avg Acc = {:.2f}'.format(args.name_src, iter_num, max_iter,
                                                                acc_s_te, acc_cls_avg_te) + '\n' + acc_list
                cur_acc = acc_cls_avg_te

            else:
                if args.trte == 'stratified':
                    # Stratified cross validation ensures the existence of every class in the validation set.
                    # Safe to use per-class average accuracy.
                    acc_s_te, acc_cls_avg_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC,
                                                          per_class_flag=True, visda_flag=False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Cls Avg Acc = {:.2f}'.format(args.name_src,
                                                                iter_num, max_iter, acc_s_te, acc_cls_avg_te)
                    cur_acc = acc_cls_avg_te
                else:
                    # Conventional cross validation may lead to the absence of certain classes in validation set,
                    # esp. when the dataset includes some very small classes, e.g., Office-Home (RSUT), DomainNet.
                    # Use overall accuracy to avoid 'nan' issue.
                    acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC,
                                          per_class_flag=False, visda_flag=False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
                    cur_acc = acc_s_te

            wandb.log({'iter_num':iter_num, 'train_loss_source':classifier_loss.item(), 'Val_Accuracy_source':acc_s_te, 'Val_Cls_Avg_Acc_Source':cur_acc})
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if cur_acc >= acc_init and iter_num >= 3 * len(dset_loaders["source_tr"]):
                # first 3 epochs: not stable yet
                acc_init = cur_acc
                if args.paral:
                    best_netF = netF.module.state_dict()
                    best_netB = netB.module.state_dict()
                    best_netC = netC.module.state_dict()
                else:
                    best_netF = netF.state_dict()
                    best_netB = netB.state_dict()
                    best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC


def test_target(args):
    dset_loaders, dsets = data_load(args)

    ## set base network
    if args.net[0:3] == 'res' or args.net[0:3] == 'vgg':
        if args.net[0:3] == 'res':
            netF = network.ResBase(res_name=args.net).cuda()
        else:
            netF = network.VGGBase(vgg_name=args.net).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck).cuda()
        if args.bias_0:   
            netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck, bias=False).cuda()
        else:
            netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck, bias=True).cuda()


        args.modelpath = args.output_dir_src + '/source_F.pt'
        netF.load_state_dict(torch.load(args.modelpath))
        args.modelpath = args.output_dir_src + '/source_B.pt'
        netB.load_state_dict(torch.load(args.modelpath))
        args.modelpath = args.output_dir_src + '/source_C.pt'
        netC.load_state_dict(torch.load(args.modelpath))
        
        if args.paral:
            netF = torch.nn.DataParallel(netF)
            netB = torch.nn.DataParallel(netB)
            netC = torch.nn.DataParallel(netC)

        netF.eval()
        netB.eval()
        netC.eval()


    if args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10' or args.dset == 'VISDA-C':
        # For VisDA, print acc of each class.
        if args.net[0:3] == 'res' or args.net[0:3] == 'vgg':
            acc, acc_list, acc_cls_avg = cal_acc(dset_loaders['test'], netF=netF, netB=netB, netC=netC,
                                                 per_class_flag=True, visda_flag=True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%, Cls Avg Acc = {:.2f}%'.format(args.trte,
                                                                args.name, acc, acc_cls_avg) + '\n' + acc_list
    else:
        # For Home, DomainNet, no need to print acc of each class.
        if args.net[0:3] == 'res' or args.net[0:3] == 'vgg':
            acc, acc_cls_avg, _ = cal_acc(dset_loaders['test'], netF=netF, netB=netB, netC=netC,
                                          per_class_flag=True, visda_flag=False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%, Cls Avg Acc = {:.2f}%'.format(args.trte,
                                                                                    args.name, acc, acc_cls_avg)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


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
        if isinstance(gpu_id, int):
            available_memory = gpu_memory_map[int(gpu_id)]
        else:
            gpu_ids = [int(gpu_id) for gpu_id in args.gpu_id.split(",")]
            for gpu_id in gpu_ids:
                available_memory = gpu_memory_map[int(gpu_id)]

        # wait unless GPU memory is more than 7000
        if available_memory < 7000:
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
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['office-home-RSUT', 'domainnet', 'VISDA-RSUT', 'VISDA-RSUT-50', 'VISDA-RSUT-10',
                                 'VISDA-Beta', 'VISDA-Tweak', 'VISDA-Knockout', 'VISDA-C', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='../result')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val', 'stratified'])
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--source_balanced', default=False, action='store_true')
    parser.add_argument('--normal_to_genhao20', default=False, action='store_true')
    parser.add_argument('--bias_0', default=False, action='store_true')
    parser.add_argument('--wandb_name', type=str, default='S', help="wandb的任务名(run中查看)")
    parser.add_argument('--wandb_new_project_name', type=str, default='', help="wandb_project_name尾缀(用来和之前的wandb_project区分)")
    parser.add_argument('--abc', default=False, action='store_true', help="abc其中已经包含了balanced_softmax,因此--abc和--balanced_softmax不能同时为true！")
    parser.add_argument('--balanced_softmax', default=False, action='store_true')  # --abc为true的话就包含了--balancedsoftmax！
    parser.add_argument('--netB_lr', type=float, default=1)
    parser.add_argument('--netF_lr', type=float, default=0.1)
    parser.add_argument('--autoaugment', default=False, action='store_true')
    parser.add_argument('--paral', default=False, action='store_true')
    
    args = parser.parse_args()


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

    folder = '../data/'
    if args.dset == 'office-home-RSUT':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'

        args.s_val_path = folder + args.dset + '/' + names[args.s] + '_BS.txt'  # val set is balanced

    elif args.dset == 'domainnet':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_train_mini.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'

        args.s_val_path = folder + args.dset + '/' + names[args.s] + '_train_mini_val.txt'   # val set is balanced


    elif args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'

        args.s_val_path = folder + args.dset + '/' + names[args.s] + '_RS_BaVal.txt'   # val set is balanced

    else:
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'

    project_name = "Imbalanced-SFDA" + "_" + args.dset + args.wandb_new_project_name
    wandb_name = names[args.s][0].upper() + "_" + args.wandb_name
    wandb.init(project = project_name, config = args.__dict__, name = wandb_name)        
    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    wait_for_GPU_avaliable(args.gpu_id)

    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):

        if i == args.s:
            continue

        if args.dset == 'office-home-RSUT' and names[i] == 'Art':
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        folder = '../data/'
        if args.dset == 'office-home-RSUT':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
        elif args.dset == 'domainnet':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_train_mini.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'
        elif args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
        else:
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'

        # test_target(args)
    wandb.finish()