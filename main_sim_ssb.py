import logging
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from utils import set_model_config, \
    set_dataset, create_model, set_parser, \
    set_seed
from trainer_sim_ssb import train

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class SimMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(SimMatch_Net, self).__init__()
        self.backbone = base
        self.feat_planes = base.channels

        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_planes, proj_size)
        ])

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def forward(self, x, **kwargs):
        logits, logits_open, feat = self.backbone(x, feature=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits': logits, 'logits_open': logits_open, 'feat': feat_proj}

    # def group_matcher(self, coarse=False):
    #     matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
    #     return matcher


def main():
    args = set_parser()
    args.start_epoch = 0
    global best_acc
    global best_acc_val

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)
    logger.info(dict(args._get_kwargs()))
    if args.seed is not None:
        set_seed(args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
    set_model_config(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_trainloader, unlabeled_dataset, test_loader, val_loader, ood_loaders \
        = set_dataset(args)

    # model, optimizer, scheduler = set_models(args)
    model = create_model(args)
    model = SimMatch_Net(model, proj_size=128)
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.opt == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
    elif args.opt == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=2e-3)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    model.zero_grad()
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler)


if __name__ == '__main__':
    main()
