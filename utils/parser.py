import argparse

__all__ = ['set_parser']


def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch OpenMatch Training')
    ## Computational Configurations
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                    help="don't use progress bar")
    parser.add_argument('--eval_only', type=int, default=0,
                        help='1 if evaluation mode ')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='for cifar10')

    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--root', default='./data', type=str,
                        help='path to data directory')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet', 'cross'],
                        help='dataset name')
    ## Hyper-parameters
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='optimize name')
    parser.add_argument('--num-labeled', type=int, default=400,
                        choices=[4, 10, 15, 25, 50, 100, 400],
                        help='number of labeled data per each class')
    parser.add_argument('--num_val', type=int, default=50,
                        help='number of validation data per each class')
    parser.add_argument('--num-super', type=int, default=10,
                        help='number of super-class known classes cifar100: 10 or 15')
    parser.add_argument('--imgnet_percent', type=int, default=10,
                        help='1% or 10% labeled data for imagenet-30')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet_imagenet'],
                        help='dataset name')
    ## HP unique to OpenMatch (Some are changed from FixMatch)
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                    help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                    help='coefficient of SOCR loss, 0.5 for CIFAR10, ImageNet, '
                         '1.0 for CIFAR100')
    parser.add_argument('--lambda_ova', default=1, type=float, help='should always be 1')
    parser.add_argument('--lambda_x', default=1, type=float, help='should always be 1')
    parser.add_argument('--lambda_u', default=1, type=float, help='should always be 1')
    parser.add_argument('--lambda_rot', default=1, type=float, help='should always be 1')
    parser.add_argument('--lambda_remix_e', default=1, type=float, help='should always be 1')
    parser.add_argument('--lambda_nrc', default=1, type=float)
    parser.add_argument('--start_fix', default=10, type=int,
                        help='epoch to start fixmatch training')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--total-steps', default=2 ** 19, type=int,
                        help='number of total steps to run')
    parser.add_argument('--epochs', default=512, type=int,
                        help='number of epochs to run')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='pseudo label threshold')
    ##
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')

    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--mix_alpha', default=0.75, type=float,
                        help='for ReMixMatch')
    parser.add_argument('--threshold_suppress', default=0.5, type=float,
                        help='lower than this, apply suppression')
    parser.add_argument('--lambda_suppress', default=1, type=float,
                        help='coefficient of suppression loss')
    parser.add_argument('--dist', default='ce', type=str, help='distance metric for suppression')
    parser.add_argument('--return_idx', action='store_true', default=False, help='return index of datasets')
    parser.add_argument('--skip_expand', action='store_true', default=False, help='skip adding labels in x_u_split')

    # neighbor hypers
    parser.add_argument('--aff_value', default=0.1, type=float, help='affinity value of nonRNN nearest neighbors')
    parser.add_argument('--nrc_warm_epoch', default=0, type=int, help='number of warmup epochs before running nrc')
    parser.add_argument('--K', default=3, type=int, help='KNN')
    parser.add_argument('--KK', default=3, type=int, help='RNN')
    parser.add_argument('--nrc_ema', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--nrc_type', default='kl_div', type=str, help='distance metric for NRC loss')
    parser.add_argument('--resume_opt', action='store_true', default=False, help='only for main_resume_neighbor')
    parser.add_argument('--nrc_DA', default='weak', type=str, help='strong DA or weak DA for nrc loss')
    parser.add_argument('--nrc_ema_model', action='store_true', default=False, help='use EMA model to update memory bank')
    parser.add_argument('--nrc_score_ema_model', action='store_true', default=False, help='use ema to compute score_near')
    parser.add_argument('--nrc_score_grad', action='store_true', default=False, help='when use online to compute score_near, allow grad')

    parser.add_argument('--nrc_DA_knn', default='weak', type=str, help='strong DA or weak DA for KNNs')
    parser.add_argument('--nrc_DA_x', default='weak', type=str, help='strong DA or weak DA for the current batch')
    parser.add_argument('--nrc_DA_feature', default='weak', type=str, help='strong DA or weak DA for features')
    parser.add_argument('--nrc_avg_feature_num', default=1, type=int, help='how many aug to avg over for computing dist')

    # dscrm hypers
    parser.add_argument('--margin', type=float, help='none for softmarginloss')
    # parser.add_argument('--dscrm_norm', default=2, type=int, help='p norm for distance computation')
    parser.add_argument('--dscrm_data', type=str, help='u for unlabeled data, l for labeled data, lu for both')
    parser.add_argument('--dscrm_augment', type=str, help='s for strong augmentation, w for weak augmentation, ws for both')
    parser.add_argument('--lambda_dscrm',  default=1.0, type=float, help='coefficient of dscrm loss')
    parser.add_argument('--dscrm_exclude_OOD', action='store_true', default=False, help='only for oracle experiment')
    parser.add_argument('--dscrm_threshold', default=0.0, type=float, help='confidence threshold for selecting unlabeled data for dscrm loss')

    # hard negative mining with augmention hypers
    parser.add_argument('--ova_neg_loss', type=str, choices=['all', 'top1', 'topk'], help='how to use negatives for ova loss')
    parser.add_argument('--ova_neg_DA', type=str, default='w', choices=['w', 's', 'ws'], help='how to use negatives for ova loss')
    parser.add_argument('--ova_pos_DA', type=str, default='w', choices=['w', 's', 'ws'], help='how to use positives for ova loss')

    # mimic test time pipeline
    parser.add_argument('--lambda_mimic',  default=1.0, type=float, help='coefficient of test mimic loss')
    parser.add_argument('--mimic_topk',  default=5, type=int, help='topN in L_mimic')

    # det pseudo
    parser.add_argument('--det_pseudo_th', default=0.9, type=float, help='coefficient of test mimic loss')

    # homo head for detection
    parser.add_argument('--homo_entropy_type', type=str, help='min or max')

    # neptune hypers
    parser.add_argument('--mode', default='async', type=str,
                        choices=['async', 'sync', 'read-only', 'debug'], help='neptune mode')
    parser.add_argument('--resume_run', type=str, help='neptune mode')

    # main_rotnet
    parser.add_argument('--lambda_rotnet', default=1, type=float, help='coefficient of rotnet loss')

    # unlabeled ova loss
    parser.add_argument('--lambda_ova_u', default=1, type=float, help='coefficient of unlabeled ova loss')
    parser.add_argument('--ova_unlabeled_threshold', type=float, help='ova_unlabeled_threshold ')
    parser.add_argument('--ova_unlabeled_cat', action='store_true', default=False, help='use logits_open_u1 and logits_open_u2, logits_open_s1 and logits_open_s2')
    parser.add_argument('--ova_unlabeled_DA', type=str, help='w, s, or ws')
    parser.add_argument('--ova_unlabeled_pos_DA', type=str, help='w, s, or ws')
    parser.add_argument('--ova_unlabeled_neg_DA', type=str, help='w, s, or ws')

    # incremental learning
    parser.add_argument('--det_head_increment',  default='linear', type=str, help='linear, mlp2, mlp3, mlp4')
    parser.add_argument('--both_head_increment', default='linear', type=str, help='linear, mlp2, mlp3, mlp4')
    parser.add_argument('--hidden_dim_increment', default=128, type=int, help='hidden_dim for MLP')
    parser.add_argument('--resume_momentum_buffer', action='store_true', default=False)

    # analysis of why clf head helps det head
    parser.add_argument('--analysis_unlabeled_percent', default=1., type=float, help='randomly select n% of selected unlabeled data for fixmatch unlabeled loss')

    # xinting's unreliable pseudo-label idea
    parser.add_argument('--alpha_t', default=20, type=float, help='we use 20% top entropy predictions')
    parser.add_argument('--compute_ent_with_unk_score', action='store_true', default=False)
    parser.add_argument('--ablate_entropy_mask', action='store_true', default=False)

    # use RotNet or SimCLR as pretraining methods
    parser.add_argument('--pretrain_method', default='none', type=str, help='rotnet or simclr')
    parser.add_argument('--pretrain_path', default='none', type=str, help='path to ckpt')

    # Why incremental learning helps
    # Analysis: increment improves test acc because the number of used data increases
    parser.add_argument('--analysis_increment_exclude_unlabeled_threshold', default=0.5, type=float, help='')

    # percentile for exclude_dataset
    parser.add_argument('--percentile', default=50, type=float, help='must be between 0 and 100 inclusive')
    parser.add_argument('--percentile_oracle_type', type=str, help='real or current')
    parser.add_argument('--percentile_oracle_path', type=str, help='path to checkpoint.pth.tar')

    # ablation: why joint training does not work?
    parser.add_argument('--joint_th_filter_epochs', default=400, type=int, help='how many epochs using th-based filter')

    # tune scheduler to alleviate performance decrease
    parser.add_argument('--no_scheduler', default=False, action='store_true', help='no lr decay in stage 2')

    # hypers used in trainer_increment_ova_neighbor_percentile_threshold
    parser.add_argument('--clf_threshold', type=float, help='pseudo label threshold')
    parser.add_argument('--det_threshold', type=float, help='pseudo label threshold')

    # hypers used in pseudo inlier
    parser.add_argument('--pseudo_inlier_threshold', type=float, help='pseudo label threshold')
    parser.add_argument('--pseudo_inlier_second_max_threshold', type=float, help='pseudo label threshold')
    parser.add_argument('--pseudo_inlier_conf_threshold', type=float, help='pseudo label threshold')
    parser.add_argument('--pseudo_inlier_unk_threshold', type=float, help='pseudo label threshold')

    # hypers used to debug imagenet N130 seed1
    parser.add_argument('--imgnet_det_lr', type=float, help='learning rate for det training')
    parser.add_argument('--det_start_epoch', type=float, help='learning rate for det training')

    # use cifar hypers for simmatch
    parser.add_argument('--simmatch_hyper', default=False, action='store_true', help='use cifar hypers')

    # for ablation of mtc detector head
    parser.add_argument('--mtc_head_strong', default=False, action='store_true', help='use cifar hypers')


    args = parser.parse_args()
    return args