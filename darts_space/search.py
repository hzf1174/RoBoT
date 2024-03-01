import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import pickle
import copy

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

from scipy.special import softmax

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable

from bayes_opt import BayesianOptimization

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from darts_space import utils
from darts_space.genotypes import *

from foresight.pruners.measures import fisher, grad_norm, grasp, snip, synflow, jacov


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--arch_path', type=str, default='data/sampled_archs.p', help='location of the data corpus')
parser.add_argument('--no_search', action='store_true',default=False, help='only apply sampling')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for search')
parser.add_argument('--batch_size', type=int, default=576, help='batch size')
parser.add_argument('--metric_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=5e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

parser.add_argument('--scale', type=float, default=1e2, help="")
parser.add_argument('--n_sample', type=int, default=60000, help='pytorch manual seed')

parser.add_argument('--total_iters', type=int, default=25, help='pytorch manual seed')
parser.add_argument('--init_portion', type=float, default=0.25, help='pytorch manual seed')
parser.add_argument('--acq', type=str, default='ucb',help='choice of bo acquisition function, [ucb, ei, poi]')
args = parser.parse_args()
args.cutout = False
args.auxiliary = False

args.save = 'darts/search-{}-{}'.format(args.save, args.dataset)
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, f'S{args.seed}-log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar10':
    NUM_CLASSES = 10
    from darts_space.model import NetworkCIFAR as Network
elif args.dataset == 'cifar100':
    NUM_CLASSES = 100
    from darts_space.model import NetworkCIFAR as Network
elif args.dataset == 'imagenet':
    NUM_CLASSES = 1000
    from darts_space.model import NetworkImageNet as Network
else:
    raise ValueError('Donot support dataset %s' % args.dataset)
    
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # load the dataset
    if 'cifar' in args.dataset:
        train_transform, valid_transform = eval("utils._data_transforms_%s" % args.dataset)(args)
        train_data = eval("dset.%s" % args.dataset.upper())(
            root=args.data, train=True, download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=4)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]),
            pin_memory=True, num_workers=4)
    elif 'imagenet' in args.dataset:
        train_queue, valid_queue = utils._get_imagenet(args)
    else:
        raise ValueError("Donot support dataset %s" % args.dataset)

    data_queues = [train_queue, valid_queue]

    metric_names = ['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacov']

    # the domain for search
    pbounds = {}
    for metric in metric_names:
        pbounds[metric] = (-1, 1)
    
    
    sampled_genos, opt_genos, sampled_metrics = get_pool(data_queues, args)
    id_to_key = {}

    id = 0
    for key in sampled_metrics:
        id_to_key[id] = key
        id += 1

    data_metrics = {}
    for metric in metric_names:
        data_metrics[metric] = []
        for key in sampled_metrics:
            data_metrics[metric].append(sampled_metrics[key][metric])

    # normalization
    for metric in metric_names:
        max_value = max(data_metrics[metric])
        min_value = min(data_metrics[metric])
        #print(metric, max_value, min_value)
        if max_value - min_value == 0:
            data_metrics[metric] = [float(i) for i in data_metrics[metric]]
        else:
            data_metrics[metric] = [(float(i) - min_value) / (max_value - min_value) for i in data_metrics[metric]]

    if not args.no_search:
        opt_archs = []
        opt_weights = [[]]
        opt_target = [-1]
        val_accs = {}
        global_var = [opt_archs, opt_weights, opt_target, val_accs]
        optimizer = BayesianOptimization(
            f               = lambda **kwargs: search(kwargs, global_var, data_metrics, metric_names,
                                                      sampled_genos, id_to_key, data_queues, args),
            pbounds         = pbounds,
            verbose=2,
            random_state    = args.seed
        )
        
        start = time.time()

        optimizer.maximize(
            init_points = 0,
            n_iter = args.total_iters - 1,
            acq = 'ucb',
        )

        opt_weights = opt_weights[0]
        opt_score_list = []
        num_arch = len(id_to_key)
        for arch in list(range(num_arch)):
            score = 0
            for metric in metric_names:
                weight = opt_weights[metric]
                if not np.isnan(data_metrics[metric][arch]):
                    score += weight * data_metrics[metric][arch]
            opt_score_list.append(score)
        score_list_order = np.flip(np.argsort(opt_score_list))

        for i in range(len(score_list_order)):
            if len(opt_archs) == args.total_iters:
                break
            arch = score_list_order[i]
            if arch not in opt_archs:
                logging.info("current opt_arch:" + str(arch))
                opt_archs.append(arch)
                opt_key = id_to_key[arch]
                opt_geno = sampled_genos[opt_key]
                opt_model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, opt_geno).cuda()
                val_acc = train(data_queues, opt_model)
                val_accs[arch] = val_acc

        opt_acc = min(val_accs) - 1
        opt_arch = 0
        for i in range(len(opt_archs)):
            arch = opt_archs[i]
            val_acc = val_accs[arch]
            if val_acc > opt_acc:
                opt_acc = val_acc
                opt_arch = arch

        opt_geno = id_to_key[opt_arch]

        logging.info('Search cost = %.2f(h)' % ((time.time() - start) / 3600, ))
        logging.info('Genotype = %s' % (opt_geno, ))


def get_pool(data_queues, args):
    size=[14 * 2, 7]
    train_queue, _ = data_queues

    if not os.path.exists(args.arch_path):
        start = time.time()

        logging.info('Start sampling architectures...')
        
        sampled_genos, opt_genos, sampled_metrics = {}, {}, {}
        
        new_weights = [np.random.random_sample(size) for _ in range(args.n_sample)]
        new_genos = [genotype(w.reshape(2, -1, size[-1])) for w in new_weights]
        new_keys = list(map(str, new_genos))
        
        sampled_genos = dict(zip(new_keys, new_genos))
        
        inputs, targets = next(iter(train_queue))
        inputs, targets = inputs[:args.metric_batch_size].cuda(), targets[:args.metric_batch_size].cuda()
        
        # compute the training-free metrics
        for i, (k, geno) in enumerate(sampled_genos.items()):
            if i % 1000 == 0:
                logging.info('Start computing the metrics for arch %06d' % (i, ))
            
            model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, geno).cuda()
            model.drop_path_prob = 0
            metric_list = compute_metrics(model, inputs, targets)
            sampled_metrics.update({k: metric_list})

        with open(args.arch_path, 'wb') as f:
            pickle.dump([sampled_genos, opt_genos, sampled_metrics], f)
            
        logging.info('Sampling cost=%.2f(h)' % ((time.time()- start) / 3600, ))
    else:
        with open(args.arch_path, 'rb') as f:
            sampled_genos, opt_genos, sampled_metrics = pickle.load(f)

    return sampled_genos, opt_genos, sampled_metrics


def search(weights, global_var, data_metrics, metric_names, sampled_genos, id_to_key, data_queues, args):
    opt_archs, opt_weights, opt_target, val_accs = global_var
    num_arch = len(id_to_key)

    score_list = []
    for arch in list(range(num_arch)):
        score = 0
        for metric in metric_names:
            weight = weights[metric]
            if not np.isnan(data_metrics[metric][arch]):
                score += weight * data_metrics[metric][arch]
        score_list.append(score)

    score_list_order = np.flip(np.argsort(score_list))

    opt_arch = score_list_order[0]
    logging.info("current opt_arch:" + str(opt_arch))
    if opt_arch not in val_accs:
        opt_archs.append(opt_arch)
        opt_key = id_to_key[opt_arch]
        opt_geno = sampled_genos[opt_key]
        opt_model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, opt_geno).cuda()
        val_acc = train(data_queues, opt_model)
        val_accs[opt_arch] = val_acc

        if val_acc >= opt_target[0]:
            opt_target[0] = val_acc
            opt_weights[0] = weights

    else:
        val_acc = val_accs[opt_arch]

        if val_acc >= opt_target[0]:
            opt_target[0] = val_acc
            opt_weights[0] = weights
        
    return val_acc


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

def compute_metrics(net, inputs, targets):
    metric_list = {}
    metric_list['fisher'] = sum_arr(fisher.compute_fisher_per_weight(copy.deepcopy(net).cuda(), inputs, targets, F.cross_entropy, "channel"))
    metric_list['grad_norm'] = sum_arr(grad_norm.get_grad_norm_arr(copy.deepcopy(net).cuda(), inputs, targets, F.cross_entropy))
    metric_list['snip'] = sum_arr(snip.compute_snip_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param", F.cross_entropy))
    metric_list['synflow'] = sum_arr(synflow.compute_synflow_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param"))
    metric_list['jacov'] = jacov.compute_jacob_cov(copy.deepcopy(net).cuda(), inputs, targets)
    metric_list['grasp'] = sum_arr(grasp.compute_grasp_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param", F.cross_entropy))
    return metric_list

def train(data_queues, model):
    train_queue, valid_queue = data_queues
    
    if 'imagenet' in args.dataset:
        criterion = utils.CrossEntropyLabelSmooth(NUM_CLASSES, args.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    model.train()
    
    for epoch in range(args.epochs):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        logging.info('epoch %d lr %e drop_prob %e', epoch, scheduler.get_last_lr()[0], model.drop_path_prob)

        for step, (input, target) in enumerate(train_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight*loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        
        scheduler.step()

    # validation
    valid_acc = infer(valid_queue, model, criterion)

    return valid_acc


def infer(valid_queue, model, criterion):
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %f', step, top1.avg)

    return top1.avg

def genotype(weights, steps=4, multiplier=4):
    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(
                W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene
        
    gene_normal = _parse(softmax(weights[0], axis=-1))
    gene_reduce = _parse(softmax(weights[1], axis=-1))

    concat = range(2+steps-multiplier, steps+2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

if __name__ == '__main__':
    main()