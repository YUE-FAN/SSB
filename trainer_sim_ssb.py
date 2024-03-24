"""
SimMatch training + SSB training
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
from tqdm import tqdm
from utils import AverageMeter, save_checkpoint, ova_ent, accuracy_open, accuracy, roc_id_ood, compute_roc
from utils import Logger
from utils import ova_loss, unlabeled_ova_neg_loss
import os

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


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
                (inputs_u_w, inputs_u_s, _), targets_u_gt, _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), targets_u_gt, _ = unlabeled_iter.next()
            try:
                (inputs_all_w, inputs_all_s, inputs_all_s2, inputs_all), targets_all_u, ind_u = unlabeled_all_iter.next()
                targets_all_u[targets_all_u >= args.num_classes] = args.num_classes
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, inputs_all_s2, inputs_all), targets_all_u, ind_u = unlabeled_all_iter.next()
                targets_all_u[targets_all_u >= args.num_classes] = args.num_classes

            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]  # 64
            num_ulb = inputs_u_w.shape[0]
            bank = mem_bank.clone().detach()

            inputs = torch.cat([inputs_x_w, inputs_x, inputs_x_s, inputs_x_s2,
                                inputs_all_w, inputs_all, inputs_all_s, inputs_all_s2], 0).to(args.device)
            targets_x = targets_x.to(args.device)
            ind_x = ind_x.to(args.device)

            outputs = model(inputs)   # [384, 55], [384, 110]
            logits, logits_open, feats = outputs['logits'], outputs['logits_open'], outputs['feat']
            ema_feats_x_lb = feats[:b_size]
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
                outputs = model(inputs_ws)  # [256, 55], [256, 110]
                logits, logits_open_fix, feats = outputs['logits'], outputs['logits_open'], outputs['feat']
                logits_u_w, logits_u_s = logits.chunk(2)
                ema_feats_x_ulb_w, feats_x_ulb_s = feats.chunk(2)

                with torch.no_grad():
                    ema_probs_x_ulb_w = F.softmax(logits_u_w, dim=-1)
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

                max_probs, targets_u = torch.max(probs_x_ulb_w.detach(), dim=-1)
                mask = max_probs.ge(args.threshold).float()
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
                in_loss = torch.zeros(1).to(args.device).mean()

            loss = args.lambda_x * Lx + args.lambda_ova * Lo + args.lambda_oem * L_oem  \
                   + args.lambda_socr * L_socr + lambda_u * L_fix + args.lambda_ova_u * Lo_u + lambda_in * in_loss
            update_bank(ema_feats_x_lb, targets_x, ind_x, mem_bank, labels_bank, ema_bank)
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

            logger_custom.set_names(['train_loss', 'train_loss_x', 'train_loss_o', 'train_loss_o_u', 'train_loss_oem',
                                     'train_loss_socr', 'train_loss_fix',
                                     'total_acc', 'Mask', 'Used_acc', 'Used OOD', 'mask_neg_percent_ova',
                                     'Used_neg_prec_ova',
                                     'Test Acc.', 'Test Loss', 'test_overall',
                                     'test_unk', 'test_roc', 'test_roc_softm', 'val_acc',
                                     'Test ROC C10', 'Test ROC C100', 'Test ROC SVHN', 'Test ROC lsun',
                                     'Test ROC imagenet',
                                     'test_unlabeled_acc', 'test_unlabeled_roc'])

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
            outputs, outputs_open = out_dict['logits'], out_dict['logits_open']  # [bs, num_class], [bs, 2 * num_class]
            outputs = F.softmax(outputs, 1)  # [bs, num_class]
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)  # [bs, 2, num_class]
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda()
            pred_close = outputs.data.max(1)[1]  # [bs,]
            unk_score = out_open[tmp_range, 0, pred_close]  # [bs,]
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

            ind_unk = unk_score > 0.5
            pred_close[ind_unk] = int(outputs.size(1))
            acc_all, unk_acc, size_unk = accuracy_open(pred_close,
                                                       targets,
                                                       num_classes=int(outputs.size(1)))
            acc.update(acc_all.item(), inputs.shape[0])
            unk.update(unk_acc, size_unk)

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = targets
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} "
                                            "acc: {acc:.3f}. "
                                            "unk: {unk:.3f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    acc=acc.avg,
                    unk=unk.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    # ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    if not val:
        roc = compute_roc(unk_all, label_all,
                          num_known=int(outputs.size(1)))
        roc_soft = compute_roc(-known_all, label_all,
                               num_known=int(outputs.size(1)))
        ind_known = np.where(label_all < int(outputs.size(1)))[0]
        id_score = unk_all[ind_known]
        logger.info("Closed acc: {:.4f}".format(top1.avg))
        logger.info("Overall acc: {:.4f}".format(acc.avg))
        logger.info("Unk acc: {:.4f}".format(unk.avg))
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
            outputs, outputs_open = out_dict['logits'], out_dict['logits_open']
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
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
