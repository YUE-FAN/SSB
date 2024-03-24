"""
FixMatch training
"""
import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import AverageMeter, save_checkpoint, test, test_ood
from utils import Logger
import os

logger = logging.getLogger(__name__)
best_acc = -1
best_acc_val = -1


def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler):
    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    used_c = AverageMeter()
    total_c = AverageMeter()
    used_ood = AverageMeter()
    end = time.time()

    model.train()

    unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                       sampler=RandomSampler(unlabeled_dataset),
                                       batch_size=args.batch_size * args.mu,
                                       num_workers=args.num_workers,
                                       drop_last=True)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    print(type(labeled_trainloader.dataset.transform))
    print(type(unlabeled_trainloader.dataset.transform))

    if args.local_rank in [-1, 0]:
        if args.resume:
            logger_custom = Logger(os.path.join(args.out, 'log.txt'), title='cifar', resume=True)
        else:
            logger_custom = Logger(os.path.join(args.out, 'log.txt'), title='cifar')
            logger_custom.set_names(['train_loss', 'train_loss_x', 'train_loss_fix',
                                     'total_acc', 'Mask', 'Used_acc', 'Used OOD',
                                     'Test Acc.', 'Test Loss', 'test_overall',
                                     'test_unk', 'test_roc', 'test_roc_softm', 'val_acc',
                                     'Test ROC C10', 'Test ROC C100', 'Test ROC SVHN', 'Test ROC lsun', 'Test ROC imagenet'])

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch1: [%d | %d] ' % (epoch + 1, args.epochs))
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x_w, _, _), targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                (inputs_x_w, _, _), targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s, _), targets_u_gt = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), targets_u_gt = unlabeled_iter.next()

            data_time.update(time.time() - end)

            b_size = inputs_x_w.shape[0]

            inputs_train = torch.cat([inputs_x_w, inputs_u_w, inputs_u_s], 0).to(args.device)
            targets_x = targets_x.to(args.device)

            logits, _ = model(inputs_train)
            logits_x = logits[:b_size]
            logits_u_w, logits_u_s = logits[b_size:].chunk(2)

            with torch.no_grad():
                pseudo_label = torch.softmax(logits_u_w, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()

                mask_probs.update(mask.mean().item())
                total_acc = targets_u.cpu().eq(targets_u_gt).float().view(-1)
                if mask.sum() != 0:
                    used_c.update(total_acc[mask != 0].mean(0).item(), mask.sum())
                    tmp = (targets_u_gt[mask != 0] == args.num_classes).float()
                    used_ood.update(tmp.mean().item())
                total_c.update(total_acc.mean(0).item())

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            L_fix = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss = Lx + L_fix

            optimizer.zero_grad()
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_fix.update(L_fix.item())

            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)

            batch_time.update(time.time() - end)
            end = time.time()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            test_unlabeled_data = copy.deepcopy(unlabeled_dataset)
            test_unlabeled_data.transform = test_loader.dataset.transform
            test_unlabeled_loader = DataLoader(test_unlabeled_data,
                                               shuffle=False, batch_size=args.batch_size,
                                               num_workers=args.num_workers, drop_last=False)
            _, test_unlabeled_acc_close, _, _, _, _, _, _, _ = test(args, test_unlabeled_loader, test_model, epoch)

            if len(val_loader) == 0:
                val_acc = 0
            else:
                val_acc = test(args, val_loader, test_model, epoch, val=True)
            test_loss, test_acc_close, test_overall, \
            test_unk, test_roc, test_roc_softm, test_id, f1_mi, f1_ma \
                = test(args, test_loader, test_model, epoch)

            ood_dataset_roc = {'cifar10': 0, 'cifar100': 0, 'svhn': 0, 'lsun': 0, 'imagenet': 0}
            for ood in ood_loaders.keys():
                roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
                logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))
                ood_dataset_roc[ood] = roc_ood

            logger_custom.append(
                [losses.avg, losses_x.avg, losses_fix.avg,
                 total_c.avg, mask_probs.avg, used_c.avg, used_ood.avg,
                 test_acc_close, test_loss, test_overall, test_unk, test_roc, test_roc_softm, val_acc,
                 ood_dataset_roc['cifar10'], ood_dataset_roc['cifar100'], ood_dataset_roc['svhn'],
                 ood_dataset_roc['lsun'], ood_dataset_roc['imagenet']])

            is_best = val_acc > best_acc_val
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                overall_valid = test_overall
                close_valid = test_acc_close
                unk_valid = test_unk
                roc_valid = test_roc
                roc_softm_valid = test_roc_softm
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if epoch + 1 in [100, 200, 300, 400, 450, 475, 500]:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc close': test_acc_close,
                    'acc overall': test_overall,
                    'unk': test_unk,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.out, filename=f'checkpoint_{epoch + 1}.pth.tar')

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'acc overall': test_overall,
                'unk': test_unk,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            logger.info('Valid overall acc: {:.3f}'.format(overall_valid))
            logger.info('Valid unk acc: {:.3f}'.format(unk_valid))
            logger.info('Valid roc: {:.3f}'.format(roc_valid))
            logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
    if args.local_rank in [-1, 0]:
        logger_custom.close()
