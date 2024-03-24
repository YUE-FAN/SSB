"""
FlexMatch training
"""
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from utils import AverageMeter, save_checkpoint, accuracy, roc_id_ood, compute_roc
from utils import Logger
from copy import deepcopy
from collections import Counter
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

    ulb_dest_len = len(unlabeled_dataset)
    selected_label = torch.ones((ulb_dest_len,), dtype=torch.long, ) * -1
    selected_label = selected_label.cuda()
    classwise_acc = torch.zeros((args.num_classes,)).cuda()

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch1: [%d | %d] ' % (epoch + 1, args.epochs))
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x_w, _, _), targets_x, _ = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                (inputs_x_w, _, _), targets_x, _ = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s, _), targets_u_gt, idx_u = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), targets_u_gt, idx_u = unlabeled_iter.next()

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
                # mask = max_probs.ge(args.threshold).float()

                # max_probs, max_idx = torch.max(torch.softmax(logits_x_ulb_w.detach(), dim=-1), dim=-1)
                select = max_probs.ge(args.threshold * (classwise_acc[targets_u] / (2. - classwise_acc[targets_u])))  # convex
                mask = select.to(max_probs.dtype)

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

            # update classwise acc
            if idx_u[select == 1].nelement() != 0:
                selected_label[idx_u[select == 1]] = targets_u[select == 1]
            classwise_acc = update_classwise_acc(classwise_acc, selected_label, ulb_dest_len, args.num_classes, True)

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


def test(args, test_loader, model, epoch, val=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc = AverageMeter()
    f1_mi = AverageMeter()
    f1_ma = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, _ = model(inputs)  # [bs, num_class], [bs, 2 * num_class]
            outputs = F.softmax(outputs, 1)  # [bs, num_class]
            known_score = outputs.max(1)[0]  # [bs,]

            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1)) #[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                # unk_all = unk_score
                known_all = known_score
                label_all = targets
            else:
                # unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} ".format(

                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    # ROC calculation
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    if not val:
        roc_soft = compute_roc(-known_all, label_all,
                               num_known=int(outputs.size(1)))
        roc = roc_soft
        ind_known = np.where(label_all < int(outputs.size(1)))[0]

        id_score = -(known_all[ind_known])
        logger.info("Closed acc: {:.4f}".format(top1.avg))
        logger.info("ROC: {:.4f}".format(roc))
        logger.info("ROC Softmax: {:.4f}".format(roc_soft))
        return losses.avg, top1.avg, acc.avg, \
               unk.avg, roc, roc_soft, id_score, f1_mi.avg, f1_ma.avg
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg


def test_ood(args, test_id, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            outputs, _ = model(inputs)
            outputs = F.softmax(outputs, 1)  # [bs, num_class]
            unk_score = -(outputs.max(1)[0])  # [bs,]
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
        if not args.no_progress:
            test_loader.close()
    # ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    roc = roc_id_ood(test_id, unk_all)

    return roc