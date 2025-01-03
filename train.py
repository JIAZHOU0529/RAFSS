from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import misc
from models.backbone import ResNet_18
from models.rafss import RAFSS
import dataset.raf as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='------Training------')
# Optimization options
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
#Device options
parser.add_argument('--gpu', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=410,
                        help='Number of labeled data')
parser.add_argument('--train-iteration', type=int, default=800,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--ema-decay', default=0.999, type=float)
#Data
parser.add_argument('--train-root', type=str, default='/mnt/b33c377d-a988-494e-860f-8149fffe7254/zz/Dataset/RAF-DB/basic/Image/train/',
                        help="root path to train data directory")
parser.add_argument('--train-rootCD', type=str, default='/mnt/b33c377d-a988-494e-860f-8149fffe7254/zz/Dataset/FER2013/train/',
                        help="root path to train CD data directory")
parser.add_argument('--test-root', type=str, default='/mnt/b33c377d-a988-494e-860f-8149fffe7254/zz/Dataset/FER2013/test/',
                        help="root path to test data directory")
parser.add_argument('--label-train', default='/mnt/b33c377d-a988-494e-860f-8149fffe7254/zz/Dataset/RAF-DB/basic/EmoLabel/train_label.txt', type=str, help='')
parser.add_argument('--label-trainCD', default='/mnt/b33c377d-a988-494e-860f-8149fffe7254/zz/Dataset/FER2013/train_label.txt', type=str, help='')
parser.add_argument('--label-test', default='/mnt/b33c377d-a988-494e-860f-8149fffe7254/zz/Dataset/FER2013/test_label.txt', type=str, help='')

                                                                                                                                                                                                                                                                                                                                                                                                        
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing Datasets')
    mean = [0.4061, 0.4521, 0.5747]
    std = [0.1858, 0.1939, 0.2116]

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_trainCD = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_labeled_set, train_labeled_set_CD, train_unlabeled_set, test_set = dataset.get_raf(args.train_root, args.label_train, args.train_rootCD, args.label_trainCD, args.test_root, args.label_test, args.n_labeled, transform_train=transform_train, transform_trainCD=transform_trainCD, transform_val=transform_val)
    print("train_label_set:", len(train_labeled_set + train_labeled_set_CD))
    print("train_label_set_CD:", len(train_labeled_set_CD))
    print("train_unlabel_set", len(train_unlabeled_set))
    print("test_set:", len(test_set))

    train_labeled_set = train_labeled_set + train_labeled_set_CD
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("labeled_trainloader", len(labeled_trainloader))
    print("unlabeled_trainloader", len(unlabeled_trainloader))
    print("test_loader:", len(test_loader))

    # Model
    print("==> creating Model")

    def create_model(ema=False):
        model = ResNet_18()
        model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)

    logger = Logger(os.path.join(args.out, 'resultRAF2CK+.txt'), title='RAF')
    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Train Loss S', 'Test Loss', 'Test Acc.'])

    test_accs = []
    threshold = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    start_epoch = 1
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):

        print('\nEpoch: [%d | %d] LR: %f Threshold=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (epoch, args.epochs, state['lr'], threshold[0], threshold[1], threshold[2], threshold[3], threshold[4], threshold[5], threshold[6]))

        train_loss, train_loss_x, train_loss_u, train_loss_sim = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, threshold, epoch, use_cuda)
        _, train_acc, outputs_new, targets_new = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        threshold = adaptive_threshold_generate(outputs_new, targets_new, threshold, epoch)

        test_loss, test_acc, _, _ = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats')

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, train_loss_sim, test_loss, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
        print('Best acc:')
        print(best_acc)
    logger.close()

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion_ce, threshold, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_sim = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)


        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            inputs_strong = inputs_strong.cuda()


        outputs_u, feature_u = model(inputs_u)
        outputs_u2, feature_u2 = model(inputs_u2)

        output_x, _ = model(inputs_x)
        # supervised loss
        Lx = criterion_ce(output_x, targets_x.long()).mean()


        # compute means of two augment for unlabeled data
        p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        max_probs, max_idx = torch.max(p, dim=1)
        max_idx = max_idx.detach()

        # contrastive loss
        # mask represents the magnitude relationship with the threshold
        mask = mask_generate(max_probs, max_idx, batch_size, threshold)
        mask_idx = np.where(mask.cpu() == 0)[0]

        # unsupervised loss
        output_strong, _ = model(inputs_strong)
        Lu = criterion_ce(output_strong, max_idx.long()) * mask
        Lu = Lu.mean()
        loss = 0.5*Lx + Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, losses_sim.avg)

def validate(valloader, model, criterion, epoch, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    outputs_new = torch.ones(1, 7).cuda()
    targets_new = torch.ones(1).long().cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets).mean()
            ##
            outputs_new = torch.cat((outputs_new, outputs), dim=0)
            targets_new = torch.cat((targets_new, targets), dim=0)
            ##
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Accuracy: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, outputs_new, targets_new)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def mask_generate(max_probs, max_idx, batch, threshold):
    mask_ori = torch.zeros(batch).cuda()
    for i in range(7):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k]
    return mask_ori.cuda()

def adaptive_threshold_generate(outputs, targets, threshold, epoch):
    outputs_l = outputs[1:, :]
    targets_l = targets[1:]
    probs = torch.softmax(outputs_l, dim=1)
    max_probs, max_idx = torch.max(probs, dim=1)

    eq_idx = np.where(targets_l.eq(max_idx).cpu() == 1)[0]
    probs_new = max_probs[eq_idx]
    targets_new = targets_l[eq_idx]

    for i in range(7):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            threshold[i] = probs_new[idx].mean().cpu() * 0.98 / (1 + math.exp( -1 * epoch)) if probs_new[idx].mean().cpu() * 0.98 / (1 + math.exp(-1 * epoch)) >= 0.8 else 0.8
        else:
            threshold[i] = 0.8
    return threshold

if __name__ == '__main__':
    main()
