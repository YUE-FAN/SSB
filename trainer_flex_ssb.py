"""
FlexMatch training + SSB training
"""

import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_2strong, TransformFixMatch_2strong
from utils import AverageMeter, save_checkpoint, ova_ent, test, test_ood
from utils import Logger
from utils import ova_loss, unlabeled_ova_neg_loss
import os
from copy import deepcopy
from collections import Counter

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp

    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_o_u = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    used_c = AverageMeter()
    total_c = AverageMeter()
    used_ood = AverageMeter()

    mask_neg_percent_ova = AverageMeter()
    used_neg_prec_ova = AverageMeter()
    end = time.time()

    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"

    model.train()
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformFixMatch_2strong
    elif args.dataset == 'cifar100' or args.dataset == 'cross':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformFixMatch_2strong
    elif 'imagenet' in args.dataset:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_2strong

    unlabeled_dataset_all.transform = func_trans(mean=mean, std=std)
    labeled_trainloader.dataset.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    if args.local_rank in [-1, 0]:
        logger_custom = Logger(os.path.join(args.out, 'log.txt'), title='cifar')
        logger_custom.set_names(['train_loss', 'train_loss_x', 'train_loss_o', 'train_loss_o_u', 'train_loss_oem',
                                 'train_loss_socr', 'train_loss_fix',
                                 'total_acc', 'Mask', 'Used_acc', 'Used OOD', 'mask_neg_percent_ova', 'Used_neg_prec_ova',
                                 'Test Acc.', 'Test Loss', 'test_overall',
                                 'test_unk', 'test_roc', 'test_roc_softm', 'val_acc',
                                 'Test ROC C10', 'Test ROC C100', 'Test ROC SVHN', 'Test ROC lsun', 'Test ROC imagenet',
                                 'test_unlabeled_acc', 'test_unlabeled_roc'])

    ulb_dest_len = len(unlabeled_dataset)
    selected_label = torch.ones((ulb_dest_len,), dtype=torch.long, ) * -1
    selected_label = selected_label.cuda()
    classwise_acc = torch.zeros((args.num_classes,)).cuda()

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        for g in optimizer.param_groups:
            print(f"lr={g['lr']}")
        output_args["epoch"] = epoch

        unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                           sampler = train_sampler(unlabeled_dataset),
                                           batch_size = args.batch_size * args.mu,
                                           num_workers = args.num_workers,
                                           drop_last = True)
        unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                           sampler=train_sampler(unlabeled_dataset_all),
                                           batch_size=args.batch_size * args.mu,
                                           num_workers=args.num_workers,
                                           drop_last=True)

        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)
        unlabeled_all_iter = iter(unlabeled_trainloader_all)


        for batch_idx in range(args.eval_step):

            try:
                (inputs_x_w, inputs_x_s, inputs_x_s2, inputs_x), targets_x, ind_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                (inputs_x_w, inputs_x_s, inputs_x_s2, inputs_x), targets_x, ind_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s, _), targets_u_gt, idx_u = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), targets_u_gt, idx_u = unlabeled_iter.next()
            try:
                (inputs_all_w, inputs_all_s, inputs_all_s2, inputs_all), targets_all_u, ind_u = unlabeled_all_iter.next()
                targets_all_u[targets_all_u >= args.num_classes] = args.num_classes
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, inputs_all_s2, inputs_all), targets_all_u, ind_u = unlabeled_all_iter.next()
                targets_all_u[targets_all_u >= args.num_classes] = args.num_classes

            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]  # 64
            inputs = torch.cat([inputs_x_w, inputs_x, inputs_x_s, inputs_x_s2,
                                inputs_all_w, inputs_all, inputs_all_s, inputs_all_s2], 0).to(args.device)
            targets_x = targets_x.to(args.device)

            logits, logits_open = model(inputs)  # [384, 55], [384, 110]
            logits_open_u1, logits_open_u2, logits_open_s1, logits_open_s2 = logits_open[4*b_size:].chunk(4)

            Lx = F.cross_entropy(logits[:2*b_size], targets_x.repeat(2), reduction='mean')
            Lo = ova_loss(args, logits_open[:2*b_size], logits_open[2*b_size:4*b_size], targets_x.repeat(2), targets_x.repeat(2))

            # unlabeled OVA loss starts
            if epoch >= args.start_fix and args.lambda_ova_u != 0:
                with torch.no_grad():
                    logits_open_w = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
                    logits_open_w = F.softmax(logits_open_w, 1)
                    know_score_w = logits_open_w[:, 1, :]  # [bs, num_class]

                    neg_mask = (know_score_w <= args.ova_unlabeled_threshold).float()  # [bs, num_class]

                    mask_neg_percent_ova.update(neg_mask.mean(dim=1).mean().item())

                    tmp = torch.zeros((neg_mask.size(0), neg_mask.size(1) + 1))  # [bs, num_class]
                    tmp.scatter_(1, targets_all_u.view(-1, 1), 1)
                    gt_mask = (1 - tmp).float()
                    gt_mask = gt_mask[:, :-1]

                    if neg_mask.cpu().view(-1).sum() != 0:
                        prec = ((neg_mask.cpu() == gt_mask) * neg_mask.cpu()).view(-1).sum() / neg_mask.cpu().view(
                            -1).sum()
                        used_neg_prec_ova.update(prec.item())

                Lo_u = unlabeled_ova_neg_loss(args, logits_open_u1, logits_open_u2, logits_open_s1, logits_open_s2, neg_mask)
            else:
                Lo_u = torch.zeros(1).to(args.device).mean()
            # unlabeled OVA loss ends

            # Open-set entropy minimization
            L_oem = ova_ent(logits_open_u1) / 2.
            L_oem += ova_ent(logits_open_u2) / 2.

            # Soft consistenty regularization
            logits_open_u1_ = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2_ = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            logits_open_u1_ = F.softmax(logits_open_u1_, 1)
            logits_open_u2_ = F.softmax(logits_open_u2_, 1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1_ - logits_open_u2_)**2, 1), 1))

            if epoch >= args.start_fix:
                inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
                logits, logits_open_fix = model(inputs_ws)  # [256, 55], [256, 110]
                logits_u_w, logits_u_s = logits.chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                select = max_probs.ge(args.threshold * (classwise_acc[targets_u] / (2. - classwise_acc[targets_u])))  # convex
                mask = select.to(max_probs.dtype)

                mask_probs.update(mask.mean().item())
                L_fix = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                total_acc = targets_u.cpu().eq(targets_u_gt).float().view(-1)
                if mask.sum() != 0:
                    used_c.update(total_acc[mask != 0].mean(0).item(), mask.sum())
                    tmp = (targets_u_gt[mask != 0] == args.num_classes).float()
                    used_ood.update(tmp.mean().item())
                total_c.update(total_acc.mean(0).item())
            else:
                L_fix = torch.zeros(1).to(args.device).mean()

            loss = args.lambda_x * Lx + args.lambda_ova * Lo + args.lambda_oem * L_oem  \
                   + args.lambda_socr * L_socr + args.lambda_u * L_fix + args.lambda_ova_u * Lo_u

            # update classwise acc
            if idx_u[select == 1].nelement() != 0:
                selected_label[idx_u[select == 1]] = targets_u[select == 1]
            classwise_acc = update_classwise_acc(classwise_acc, selected_label, ulb_dest_len, args.num_classes, True)

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_o_u.update(Lo_u.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_oem"] = losses_oem.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]


            optimizer.step()
            if (args.opt != 'adam') and (not args.no_scheduler):
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            test_unlabeled_data = copy.deepcopy(unlabeled_dataset_all)
            test_unlabeled_data.transform = test_loader.dataset.transform
            test_unlabeled_data.return_idx = False
            test_unlabeled_loader = DataLoader(test_unlabeled_data,
                                               shuffle=False, batch_size=args.batch_size,
                                               num_workers=args.num_workers, drop_last=False)
            _, test_unlabeled_acc_close, _, _, test_unlabeled_roc, _, _, _, _ = test(args, test_unlabeled_loader,
                                                                                     test_model, epoch)

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
                [losses.avg, losses_x.avg, losses_o.avg, losses_o_u.avg, losses_oem.avg, losses_socr.avg, losses_fix.avg,
                 total_c.avg, mask_probs.avg, used_c.avg, used_ood.avg, mask_neg_percent_ova.avg, used_neg_prec_ova.avg,
                 test_acc_close, test_loss, test_overall, test_unk, test_roc, test_roc_softm, val_acc,
                 ood_dataset_roc['cifar10'], ood_dataset_roc['cifar100'], ood_dataset_roc['svhn'],
                 ood_dataset_roc['lsun'], ood_dataset_roc['imagenet'],
                 test_unlabeled_acc_close, test_unlabeled_roc])

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


@torch.no_grad()
def update_classwise_acc(classwise_acc, selected_label, ulb_dest_len, num_classes, thresh_warmup):
    pseudo_counter = Counter(selected_label.tolist())
    if max(pseudo_counter.values()) < ulb_dest_len:  # not all(5w) -1
        if thresh_warmup:
            for i in range(num_classes):
                classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
        else:
            wo_negative_one = deepcopy(pseudo_counter)
            if -1 in wo_negative_one.keys():
                wo_negative_one.pop(-1)
            for i in range(num_classes):
                classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
    return classwise_acc
