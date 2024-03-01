import os
import sys
import time
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from darts_space import genotypes
from darts_space.model import NetworkImageNet as Network
from darts_space import utils
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

def get_args():
    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument("--data_dir", type=str, default="./data/imagenet", help="location of ImageNet")
    parser.add_argument("--class_num", type=int, default=1000, help="the number of classes")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="init learning rate")
    parser.add_argument("--learning_rate_min", type=float, default=0.0, help="minimal init learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear, exp or cosine')
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="weight decay")
    parser.add_argument("--report_freq", type=float, default=100, help="report frequency")
    parser.add_argument("--epochs", type=int, default=250, help="num of training epochs")
    parser.add_argument("--start_epoch", type=int, default=0, help="current epoch for reload")
    parser.add_argument("--init_channels", type=int, default=48, help="num of init channels")
    parser.add_argument("--layers", type=int, default=14, help="total number of layers")
    parser.add_argument("--auxiliary", action="store_true", default=False, help="use auxiliary tower")
    parser.add_argument("--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss")
    parser.add_argument("--drop_path_prob", type=float, default=0.0, help="drop path probability")
    parser.add_argument("--save", type=str, default="ImageNet", help="experiment name")
    parser.add_argument("--reload", action="store_true", default=False, help="reload initial weights")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--arch", type=str, default="NASI", help="which architecture to use")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--tensorboard_dir', default='./tensorboard/', type=str, help='tensorboard log')
    #parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()
    return args

args = get_args()
args.local_rank = int(os.environ['LOCAL_RANK'])
args.gpu_num = torch.cuda.device_count()
if args.local_rank==0: writer = SummaryWriter(f'{os.path.expanduser(args.tensorboard_dir)}{args.save}/')
args.save = f'eval/{args.save}'
if args.local_rank==0: utils.create_exp_dir(args.save, scripts_to_save=None)
else: time.sleep(1)
logging = utils.logger(os.path.join(args.save, f"log_{args.local_rank}.txt"), True, args.local_rank==0)

def main():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)

    logging.info("args = %s", args)

    train_queue, valid_queue, sampler = utils._get_dist_imagenet(args)
    logging.info(f'rank {args.local_rank}: load data successfully')

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, args.class_num, args.layers, args.auxiliary, genotype)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.local_rank)
    criterion_smooth = utils.CrossEntropyLabelSmooth(args.class_num, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda(args.local_rank)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # default 'cosine' ('linear' direct manipulate lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    if args.lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

    best_acc = 0.
    if args.reload:
        temp_filename = os.path.join(args.save, 'model_best.pth.tar')
        ckpt = torch.load(temp_filename)
        args.start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        torch.set_rng_state(ckpt['seed'])
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    lr = args.learning_rate
    for epoch in range(args.start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        # lr
        if args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            current_lr = scheduler.get_last_lr()[0]
        # lr warmup for 5 epochs
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        else:
            logging.info("epoch %d lr %e", epoch, current_lr)

        # drop_path
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc_1, train_acc_5, train_obj = run(epoch, train_queue, model, criterion_smooth, optimizer)
        logging.info(f"train_acc_1: {train_acc_1:8.5f}\ttrain_acc_5: {train_acc_5:8.5f}")

        # lr step for cosine/exp
        if args.lr_scheduler != 'linear':
            scheduler.step()

        # eval & checkpoint only on GPU 0
        if args.local_rank == 0:
            valid_acc_1, valid_acc_5, valid_obj = run(epoch, valid_queue, model, criterion)
            logging.info(f"valid_acc_1: {valid_acc_1:8.5f}\tvalid_acc_5: {valid_acc_5:8.5f}")
            if valid_acc_1 > best_acc:
                best_acc = valid_acc_1
                logging.info("best_acc %f", best_acc)
            # checkpoint
            utils.save_checkpoint({'epoch': epoch,
                             'best_acc': best_acc,
                             'seed': torch.get_rng_state(),
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict()}, valid_acc_1>=best_acc, args.save)

def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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

        if args.local_rank == 0:
            # tensorboardX
            writer.add_scalar(f'{namespace}/step-loss', objs.avg, step + len(data_loader) * epoch)
            writer.add_scalar(f'{namespace}/step-top1', top1.avg, step + len(data_loader) * epoch)
            writer.add_scalar(f'{namespace}/step-top5', top5.avg, step + len(data_loader) * epoch)

        if step % args.report_freq == 0:
            logging.info(f'{namespace} {step:3d}/{len(data_loader)} '
                         f'{time_avg.avg:6.3f} {objs.avg:9.6f} {top1.avg:9.6f} {top5.avg:9.6f}')

        time_avg.update(time.time() - timestamp)
        timestamp = time.time()

    if args.local_rank == 0:
        writer.add_scalar(f'{namespace}/epoch-top1', top1.avg, epoch)
        writer.add_scalar(f'{namespace}/epoch-top5', top5.avg, epoch)
        writer.add_scalar(f'{namespace}/epoch-loss', objs.avg, epoch)
        writer.flush()
    return top1.avg, top5.avg, objs.avg


if __name__ == "__main__":
    main()
