import os
import sys
import time
import numpy as np
import torch
import logging
import argparse

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ImportWarning)

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import darts_space.genotypes as genotypes
from darts_space.model import NetworkCIFAR as Network
from darts_space import utils
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser("cifar")
parser.add_argument("--data", type=str, default="data", help="location of the data corpus")
parser.add_argument("--dataset", type=str, default="cifar10", help="[cifar10, cifar100]")
parser.add_argument("--batch_size", type=int, default=96, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
parser.add_argument("--learning_rate_min", type=float, default=0.0, help="final learning rate")
parser.add_argument("--gamma", type=float, default=0.995, help="final learning rate")
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler, linear, exp or cosine')
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--epochs", type=int, default=600, help="num of training epochs")
parser.add_argument("--init_channels", type=int, default=36, help="num of init channels")
parser.add_argument("--layers", type=int, default=20, help="total number of layers")
parser.add_argument("--auxiliary", action="store_true", default=False, help="use auxiliary tower")
parser.add_argument("--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument("--drop_path_prob", type=float, default=0.2, help="drop path probability")
parser.add_argument("--save", type=str, default="HNAS", help="experiment name")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--arch", type=str, default="HNAS1", help="which architecture to use")
parser.add_argument("--gpu", type=int, default=0, help="gpu to use")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument('--tensorboard_dir', default='./tensorboard/', type=str, help='tensorboard log')
args = parser.parse_args()

if args.dataset == 'cifar10':
    args.class_num = 10
    args.save = f'C10-{args.save}'
elif args.dataset == 'cifar100':
    args.class_num = 100
    args.save = f'C100-{args.save}'
else:
    raise ValueError('Donot support dataset %s' %(args.dataset))

writer = SummaryWriter(f'{os.path.expanduser(args.tensorboard_dir)}{args.save}-{args.arch}-{args.seed}/')
args.save = f'eval/{args.save}-{args.arch}-{args.seed}'
utils.create_exp_dir(args.save, scripts_to_save=None)
logging = utils.logger(os.path.join(args.save, "log.txt"), True, True)

def main():
    torch.cuda.set_device(args.gpu)

    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)

    train_queue, valid_queue = eval("utils._get_%s" %(args.dataset))(args)
    logging.info("load data successfully")

    genotype = eval("genotypes.%s" % args.arch)
    
    model = Network(args.init_channels, args.class_num, args.layers, args.auxiliary, genotype)
    model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), 
            eta_min=args.learning_rate_min
        )
    elif args.lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.gamma, 
        )
    elif args.lr_scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda = lambda epoch: (args.learning_rate - (args.learning_rate - args.learning_rate_min) * (epoch + 1) / args.epochs) / (args.learning_rate - (args.learning_rate - args.learning_rate_min) * epoch / args.epochs)
        )
    else:
        raise ValueError('Donot support learning rate scheduler %s' % args.lr_scheduler)

    for epoch in range(args.epochs):
        logging.info("epoch %d lr %e", epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = run(epoch, train_queue, model, criterion, optimizer)
        logging.info("train_acc %f", train_acc)

        valid_acc, valid_obj = run(epoch, valid_queue, model, criterion)
        logging.info("valid_acc %f", valid_acc)

        utils.save(model, os.path.join(args.save, "weights.pt"))
        scheduler.step()


def run(epoch, data_loader, model, criterion, optimizer=None):
    if optimizer:
        model.train()
        namespace = 'train'
    else:
        model.eval()
        namespace = 'valid'

    time_avg = utils.AvgrageMeter()
    objs, top1, top5 = utils.AvgrageMeter(), utils.AvgrageMeter(), utils.AvgrageMeter()

    timestamp = time.time()
    for step, (input, target) in enumerate(data_loader):
        input = input.cuda()
        target = target.cuda()

        if optimizer:
            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                logits, _ = model(input)
                loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        writer.add_scalar(f'{namespace}/step-loss', objs.avg, step + len(data_loader) * epoch)
        writer.add_scalar(f'{namespace}/step-top1', top1.avg, step + len(data_loader) * epoch)
        writer.add_scalar(f'{namespace}/step-top5', top5.avg, step + len(data_loader) * epoch)

        if step % args.report_freq == 0:
            logging.info(f'{namespace} {step:3d}/{len(data_loader)} '
                         f'{time_avg.avg:6.3f} {objs.avg:9.6f} {top1.avg:9.6f} {top5.avg:9.6f}')

        time_avg.update(time.time() - timestamp)
        timestamp = time.time()

    writer.add_scalar(f'{namespace}/epoch-top1', top1.avg, epoch)
    writer.add_scalar(f'{namespace}/epoch-top5', top5.avg, epoch)
    writer.flush()
    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
