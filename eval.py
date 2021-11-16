import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
import numpy as np
import random

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch RBNN Training')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
parser.add_argument('--load', metavar='LOAD',
                    default='result', help='loaded folder')
parser.add_argument('--datasets', metavar='DATASETS', type=list, nargs='+',
                    default=['cifar10', 'fashionmnist', 'svhn'],
                    help="['cifar10', 'fashionmnist', 'svhn', 'cifar100']")
parser.add_argument('--model', '-a', metavar='MODEL', default='reactnet', choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: reactnet)')
parser.add_argument('--input_size', type=int, default=32, help='image input size')
parser.add_argument('--type', default='torch.cuda.FloatTensor', help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0', help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', type=int, default=0, help='initial seed for randomness')


def forward_eval(data_loader, model, criterion, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_ = [0] * len(data_loader)
    top1_ = [0] * len(data_loader)
    top5_ = [0] * len(data_loader)
    end = time.time()
    for i in range(len(data_loader)):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for batch_idx, (inputs, target) in enumerate(data_loader[i]):
            # measure data loading time
            data_time.update(time.time() - end)
            if args.gpus is not None:
                target = target.cuda()

            with torch.no_grad():
                input_var = Variable(inputs.type(args.type), volatile=True)
                target_var = Variable(target)
                # refactoring weights in order to switch tasks
                model.reset_seed(2021 * i + args.seed)
                logsoftmax = nn.LogSoftmax(1)
                output = logsoftmax(model(input_var))[:, :num_classes[i]]

            loss = criterion(output, target_var)

            # measure accuracy and record loss
            top_max = 5 if num_classes[i] >= 5 else num_classes[i]
            prec1, prec5 = accuracy(output.data, target, topk=(1, top_max))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('{phase} - [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f}\t'
                      'Prec@1 {top1.val:.3f}\t'
                      'Prec@5 {top5.val:.3f}'.format(
                    batch_idx, len(data_loader[i]),
                    phase='EVALUATING',
                    batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        losses_[i] = losses.avg
        top1_[i] = top1.avg
        top5_[i] = top5.avg

    return losses_, top1_, top5_


def validate(data_loader, model, criterion, num_classes):
    model.eval()
    return forward_eval(data_loader, model, criterion, num_classes)


def generate_dataloader(model):
    trainloaders = []
    testloaders = []
    for dataset in args.datasets:
        default_transform = {
            'train': get_transform(dataset,
                                   input_size=args.input_size, augment=True),
            'test': get_transform(dataset,
                                  input_size=args.input_size, augment=False)
        }

        transform = getattr(model, 'input_transform', default_transform)
        train_data = get_dataset(dataset, 'train', transform['train'])
        val_data = get_dataset(dataset, 'test', transform['test'])

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        trainloaders.append(train_loader)
        testloaders.append(val_loader)

    return trainloaders, testloaders


def main():
    global args, best_prec1
    best_prec1 = []
    args = parser.parse_args()

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    num_classes = []
    max_num_classes = 0
    for dataset in args.datasets:
        if dataset == 'svhn':
            num_classes.append(10)
            if max_num_classes < 10:
                max_num_classes = 10
        elif 'mnist' in dataset:
            num_classes.append(10)
            if max_num_classes < 10:
                max_num_classes = 10
        elif dataset == 'cifar10':
            num_classes.append(10)
            if max_num_classes < 10:
                max_num_classes = 10
        elif dataset == 'cifar100':
            num_classes.append(100)
            if max_num_classes < 100:
                max_num_classes = 100

    # create model
    print("creating model ", args.model)
    model = models.__dict__[args.model]
    model_config = {'num_classes': max_num_classes, 'datasets': args.datasets}
    model = model(**model_config)
    print("created model with configuration: ", model_config)

    # Data loading code
    train_loader, val_loader = generate_dataloader(model)
    # Initialize the seed used to switch tasks
    for i in range(len(train_loader)):
        model.reset_seed(2021 * i + args.seed)

    load_path = os.path.join(args.results_dir, args.load) + '/model_best.pth.tar'
    checkpoint = torch.load(load_path)
    print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    # evaluate on validation set
    val_loss, val_prec1, val_prec5 = validate(
        val_loader, model, criterion, num_classes)

    for i in range(len(args.datasets)):
        print('Dataset: {0}\t'
              'Validation Loss {val_loss:.4f} \t'
              'Validation Prec@1 {val_prec1:.3f} \t'
              'Validation Prec@5 {val_prec5:.3f}'
              .format(args.datasets[i], val_loss=val_loss[i], val_prec1=val_prec1[i], val_prec5=val_prec5[i]))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(0)
    main()
    exit("exit")
