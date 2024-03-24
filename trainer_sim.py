"""
SimMatch training
"""
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from utils import AverageMeter, save_checkpoint, roc_id_ood, compute_roc, accuracy, accuracy_open
from utils import Logger
import os

logger = logging.getLogger(__name__)
best_acc = -1
best_acc_val = -1


@torch.no_grad()
def update_bank(k, labels, index, mem_bank, labels_bank, ema_bank):
    mem_bank[:, index] = F.normalize(ema_bank * mem_bank[:, index] + (1 - ema_bank) * k.t().detach())
    labels_bank[index] = labels.detach()


class DistAlignQueueHook(object):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """

    def __init__(self, num_classes, queue_length=128, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.queue_length = queue_length

        # p_target
        self.p_target_ptr, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target.mean(dim=0))
        # p_model
        self.p_model = torch.zeros(self.queue_length, self.num_classes, dtype=torch.float)
        self.p_model_ptr = torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def dist_align(self, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target.mean(dim=0) + 1e-6) / (self.p_model.mean(dim=0) + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb):
        # TODO: think better way?
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)
            if self.p_target_ptr is not None:
                self.p_target_ptr = self.p_target_ptr.to(probs_x_ulb.device)

        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
            self.p_model_ptr = self.p_model_ptr.to(probs_x_ulb.device)

        probs_x_ulb = probs_x_ulb.detach()
        p_model_ptr = int(self.p_model_ptr)
        self.p_model[p_model_ptr] = probs_x_ulb.mean(dim=0)
        self.p_model_ptr[0] = (p_model_ptr + 1) % self.queue_length

        if self.p_target_ptr is not None:
            assert probs_x_lb is not None
            p_target_ptr = int(self.p_target_ptr)
            self.p_target[p_target_ptr] = probs_x_lb.mean(dim=0)
            self.p_target_ptr[0] = (p_target_ptr + 1) % self.queue_length

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        p_target_ptr = None
        if p_target_type == 'uniform':
            p_target = torch.ones(self.queue_length, self.num_classes, dtype=torch.float) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.zeros((self.queue_length, self.num_classes), dtype=torch.float)
            p_target_ptr = torch.zeros(1, dtype=torch.long)
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
            p_target = p_target.unsqueeze(0).repeat((self.queue_length, 1))

        return p_target_ptr, p_target


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

    if args.simmatch_hyper:
        ema_bank = 0.7
        lambda_in = 1.0
        lambda_u = 1.0
        T = 0.1
        # p_cutoff = 0.95
        proj_size = 128
        K = len(labeled_trainloader.dataset)
        smoothing_alpha = 0.9
        da_len = 32
    elif 'imagenet' in args.dataset:
        ema_bank = 0.999
        lambda_in = 5.0
        lambda_u = 10.0
        T = 0.1
        args.threshold = 0.7
        proj_size = 128
        K = len(labeled_trainloader.dataset)
        smoothing_alpha = 0.9
        da_len = 256
    else:
        ema_bank = 0.7
        lambda_in = 1.0
        lambda_u = 1.0
        T = 0.1
        # p_cutoff = 0.95
        proj_size = 128
        K = len(labeled_trainloader.dataset)
        smoothing_alpha = 0.9
        da_len = 32


    dist_align = DistAlignQueueHook(num_classes=args.num_classes, queue_length=da_len, p_target_type='uniform')

    mem_bank = torch.randn(proj_size, K).to(args.device)
    mem_bank = F.normalize(mem_bank, dim=0)
    labels_bank = torch.zeros(K, dtype=torch.long).to(args.device)

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch1: [%d | %d] ' % (epoch + 1, args.epochs))
        for batch_idx in range(args.eval_step):
            try:
                (x_lb, _, _), y_lb, idx_lb = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                (x_lb, _, _), y_lb, idx_lb = labeled_iter.next()
            try:
                (x_ulb_w, x_ulb_s, _), targets_u_gt, _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (x_ulb_w, x_ulb_s, _), targets_u_gt, _ = unlabeled_iter.next()

            data_time.update(time.time() - end)

            x_lb = x_lb.to(args.device)
            x_ulb_w = x_ulb_w.to(args.device)
            x_ulb_s = x_ulb_s.to(args.device)
            y_lb = y_lb.to(args.device)
            idx_lb = idx_lb.to(args.device)

            num_lb = y_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]

            # inference and calculate sup/unsup losses
            bank = mem_bank.clone().detach()

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            outputs = model(inputs)
            logits, feats = outputs['logits'], outputs['feat']
            logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
            ema_logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            ema_feats_x_ulb_w, feats_x_ulb_s = feats[num_lb:].chunk(2)

            sup_loss = F.cross_entropy(logits_x_lb, y_lb, reduction='mean')

            with torch.no_grad():
                ema_probs_x_ulb_w = F.softmax(ema_logits_x_ulb_w, dim=-1)
                ema_probs_x_ulb_w = dist_align.dist_align(probs_x_ulb=ema_probs_x_ulb_w.detach())

            with torch.no_grad():
                teacher_logits = ema_feats_x_ulb_w @ bank
                teacher_prob_orig = F.softmax(teacher_logits / T, dim=1)
                factor = ema_probs_x_ulb_w.gather(1, labels_bank.expand([num_ulb, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if smoothing_alpha < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, args.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, labels_bank.expand([bs, -1]),
                                                                  teacher_prob_orig)
                    probs_x_ulb_w = ema_probs_x_ulb_w * smoothing_alpha + aggregated_prob * (
                                1 - smoothing_alpha)
                else:
                    probs_x_ulb_w = ema_probs_x_ulb_w

            student_logits = feats_x_ulb_s @ bank
            student_prob = F.softmax(student_logits / T, dim=1)
            in_loss = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
            if epoch == 0:
                in_loss *= 0.0
                probs_x_ulb_w = ema_probs_x_ulb_w

            # compute mask
            with torch.no_grad():
                max_probs, targets_u = torch.max(probs_x_ulb_w, dim=-1)
                mask = max_probs.ge(args.threshold).float()

                mask_probs.update(mask.mean().item())
                total_acc = targets_u.cpu().eq(targets_u_gt).float().view(-1)
                if mask.sum() != 0:
                    used_c.update(total_acc[mask != 0].mean(0).item(), mask.sum())
                    tmp = (targets_u_gt[mask != 0] == args.num_classes).float()
                    used_ood.update(tmp.mean().item())
                total_c.update(total_acc.mean(0).item())
            unsup_loss = (F.cross_entropy(logits_x_ulb_s, targets_u, reduction='none') * mask).mean()

            loss = sup_loss + lambda_u * unsup_loss + lambda_in * in_loss

            update_bank(ema_feats_x_lb, y_lb, idx_lb, mem_bank, labels_bank, ema_bank)

            optimizer.zero_grad()
            loss.backward()

            losses.update(loss.item())
            losses_x.update(sup_loss.item())
            losses_fix.update(unsup_loss.item())

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
            out_dict = model(inputs)  # [bs, num_class], [bs, 2 * num_class]
            outputs = out_dict['logits']
            outputs = F.softmax(outputs, 1)  # [bs, num_class]
            known_score = outputs.max(1)[0]  # [bs,]

            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1))#[0]
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
            out_dict = model(inputs)
            outputs = out_dict['logits']
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
