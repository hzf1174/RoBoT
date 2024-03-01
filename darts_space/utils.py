import os
import sys
import time
import warnings
import numpy as np
import torch
import shutil
import logging
import random

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

warnings.filterwarnings("error")


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
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


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _get_cifar10(args):
    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR10(
        root=args.data, train=False, download=True, transform=valid_transform
    )

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return train_queue, valid_queue


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    CIFAR_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _get_cifar100(args):
    train_transform, valid_transform = _data_transforms_cifar100(args)
    train_data = dset.CIFAR100(
        root=args.data, train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR100(
        root=args.data, train=False, download=True, transform=valid_transform
    )

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return train_queue, valid_queue


def _get_imagenet(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(args.train_portion * num_train))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=4,pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:2*split]),
        num_workers=4,pin_memory=True, drop_last=True)

    return train_loader, val_loader

def _get_dist_imagenet(args):
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.gpu_num, rank=args.local_rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size//args.gpu_num, num_workers=max(args.gpu_num*2, 4),
        pin_memory=True, drop_last=True, sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader, sampler



def count_parameters_in_MB(model):
    return np.sum([np.prod(v.size()) for n, v in model.named_parameters() if 'auxiliary' not in n])/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob, dims=(0,)):
    var_size = [1 for _ in range(x.dim())]
    for i in dims:
        var_size[i] = x.size(i)
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            *var_size).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'tools'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'tools', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class Performance(object):
    def __init__(self, path):
        self.path = path
        self.data = None

    def update(self, alphas_normal, alphas_reduce, val_loss):
        a_normal = F.softmax(alphas_normal, dim=-1)
        # print("alpha normal size: ", a_normal.data.size())
        a_reduce = F.softmax(alphas_reduce, dim=-1)
        # print("alpha reduce size: ", a_reduce.data.size())
        data = np.concatenate([a_normal.data.reshape(-1),
                               a_reduce.data.reshape(-1),
                               np.array([val_loss.data])]).reshape(1, -1)
        if self.data is not None:
            self.data = np.concatenate([self.data, data], axis=0)
        else:
            self.data = data

    def save(self):
        np.save(self.path, self.data)


def logger(log_dir, need_time=True, need_stdout=False):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y-%I:%M:%S')
    if need_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        log.addHandler(ch)
    if need_time:
        fh.setFormatter(formatter)
        if need_stdout:
            ch.setFormatter(formatter)
    log.addHandler(fh)
    return log

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss