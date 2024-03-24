# CIFAR10 with 25 labels
CUDA_VISIBLE_DEVICES=0 python main_flex.py --dataset cifar10 --num-labeled 25 --out ./saved_models/cifar10/flexmatch/wresnetleaky2_C6_N25_mlp3_head1024_seed0 --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 0.5 --batch-size 64 --mu 2 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --no-progress --both_head_increment mlp3 --threshold 0.95 --hidden_dim_increment 1024 --return_idx
CUDA_VISIBLE_DEVICES=0 python main_flex_ssb.py --dataset cifar10 --num-labeled 25 --resume ./saved_models/cifar10/flexmatch/wresnetleaky2_C6_N25_mlp3_head1024_seed0/checkpoint_475.pth.tar --out ./saved_models/cifar10/ours_flexmatch/wresnetleaky2_C6_N25_mlp3_head1024_ova_ckpt475_clf_threshold_ovaTh001_seed0/ --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 0.5 --batch-size 64 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --mu 2 --no-progress --both_head_increment mlp3 --hidden_dim_increment 1024 --return_idx --ova_neg_loss all --ova_neg_DA ws --ova_unlabeled_threshold 0.01 --ova_unlabeled_neg_DA s --threshold 0.95

# CIFAR10 with 50 labels
CUDA_VISIBLE_DEVICES=0 python main_flex.py --dataset cifar10 --num-labeled 50 --out ./saved_models/cifar10/flexmatch/wresnetleaky2_C6_N50_mlp3_head1024_seed0 --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 0.5 --batch-size 64 --mu 2 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --no-progress --both_head_increment mlp3 --threshold 0.95 --hidden_dim_increment 1024 --return_idx
CUDA_VISIBLE_DEVICES=0 python main_flex_ssb.py --dataset cifar10 --num-labeled 50 --resume ./saved_models/cifar10/flexmatch/wresnetleaky2_C6_N50_mlp3_head1024_seed0/checkpoint_475.pth.tar --out ./saved_models/cifar10/ours_flexmatch/wresnetleaky2_C6_N50_mlp3_head1024_ova_ckpt475_clf_threshold_ovaTh001_seed0/ --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 0.5 --batch-size 64 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --mu 2 --no-progress --both_head_increment mlp3 --hidden_dim_increment 1024 --return_idx --ova_neg_loss all --ova_neg_DA ws --ova_unlabeled_threshold 0.01 --ova_unlabeled_neg_DA s --threshold 0.95

# CIFAR100 with 55 known classes and 25 labels
CUDA_VISIBLE_DEVICES=0 python main_flex.py --dataset cifar100 --num-super 10 --num-labeled 25 --out ./saved_models/cifar100/flexmatch/wresnetleaky2_C55_N25_mlp3_head1024_seed0 --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --mu 2 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --no-progress --both_head_increment mlp3 --threshold 0.95 --hidden_dim_increment 1024 --return_idx
CUDA_VISIBLE_DEVICES=0 python main_flex_ssb.py --dataset cifar100 --num-super 10 --num-labeled 25 --resume ./saved_models/cifar100/flexmatch/wresnetleaky2_C55_N25_mlp3_head1024_seed0/checkpoint_475.pth.tar --out ./saved_models/cifar100/ours_flexmatch/wresnetleaky2_C55_N25_mlp3_head1024_ova_ckpt475_clf_threshold_ovaTh001_seed0/ --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --mu 2 --no-progress --both_head_increment mlp3 --hidden_dim_increment 1024 --return_idx --ova_neg_loss all --ova_neg_DA ws --ova_unlabeled_threshold 0.01 --ova_unlabeled_neg_DA s --threshold 0.95

# CIFAR100 with 55 known classes and 50 labels
CUDA_VISIBLE_DEVICES=0 python main_flex.py --dataset cifar100 --num-super 10 --num-labeled 50 --out ./saved_models/cifar100/flexmatch/wresnetleaky2_C55_N50_mlp3_head1024_seed0 --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --mu 2 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --no-progress --both_head_increment mlp3 --threshold 0.95 --hidden_dim_increment 1024 --return_idx
CUDA_VISIBLE_DEVICES=0 python main_flex_ssb.py --dataset cifar100 --num-super 10 --num-labeled 50 --resume ./saved_models/cifar100/flexmatch/wresnetleaky2_C55_N50_mlp3_head1024_seed0/checkpoint_475.pth.tar --out ./saved_models/cifar100/ours_flexmatch/wresnetleaky2_C55_N50_mlp3_head1024_ova_ckpt475_clf_threshold_ovaTh001_seed0/ --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --mu 2 --no-progress --both_head_increment mlp3 --hidden_dim_increment 1024 --return_idx --ova_neg_loss all --ova_neg_DA ws --ova_unlabeled_threshold 0.01 --ova_unlabeled_neg_DA s --threshold 0.95

# CIFAR100 with 80 known classes and 25 labels
CUDA_VISIBLE_DEVICES=0 python main_flex.py --dataset cifar100 --num-super 15 --num-labeled 25 --out ./saved_models/cifar100/flexmatch/wresnetleaky2_C80_N25_mlp3_head1024_seed0 --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --mu 2 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --no-progress --both_head_increment mlp3 --threshold 0.95 --hidden_dim_increment 1024 --return_idx
CUDA_VISIBLE_DEVICES=0 python main_flex_ssb.py --dataset cifar100 --num-super 15 --num-labeled 25 --resume ./saved_models/cifar100/flexmatch/wresnetleaky2_C80_N25_mlp3_head1024_seed0/checkpoint_475.pth.tar --out ./saved_models/cifar100/ours_flexmatch/wresnetleaky2_C80_N25_mlp3_head1024_ova_ckpt475_clf_threshold_ovaTh001_seed0/ --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --mu 2 --no-progress --both_head_increment mlp3 --hidden_dim_increment 1024 --return_idx --ova_neg_loss all --ova_neg_DA ws --ova_unlabeled_threshold 0.01 --ova_unlabeled_neg_DA s --threshold 0.95

# CIFAR100 with 80 known classes and 50 labels
CUDA_VISIBLE_DEVICES=0 python main_flex.py --dataset cifar100 --num-super 15 --num-labeled 50 --out ./saved_models/cifar100/flexmatch/wresnetleaky2_C80_N50_mlp3_head1024_seed0 --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --mu 2 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --no-progress --both_head_increment mlp3 --threshold 0.95 --hidden_dim_increment 1024 --return_idx
CUDA_VISIBLE_DEVICES=0 python main_flex_ssb.py --dataset cifar100 --num-super 15 --num-labeled 50 --resume ./saved_models/cifar100/flexmatch/wresnetleaky2_C80_N50_mlp3_head1024_seed0/checkpoint_475.pth.tar --out ./saved_models/cifar100/ours_flexmatch/wresnetleaky2_C80_N50_mlp3_head1024_ova_ckpt475_clf_threshold_ovaTh001_seed0/ --arch wideresnet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 1.0 --batch-size 64 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --mu 2 --no-progress --both_head_increment mlp3 --hidden_dim_increment 1024 --return_idx --ova_neg_loss all --ova_neg_DA ws --ova_unlabeled_threshold 0.01 --ova_unlabeled_neg_DA s --threshold 0.95

# ImageNet with 5% labels
CUDA_VISIBLE_DEVICES=0 python main_flex.py --dataset imagenet --imgnet_percent 5 --out ./saved_models/imagenet30/flexmatch/resnet18_C20_N65_mlp3_head4096 --arch resnet_imagenet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 0.5 --batch-size 64 --mu 2 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --no-progress --both_head_increment mlp3 --threshold 0.95 --hidden_dim_increment 4096 --return_idx
CUDA_VISIBLE_DEVICES=0,1 python main_flex_ssb_multigpu.py --dataset imagenet --imgnet_percent 5 --resume ./saved_models/imagenet30/flexmatch/resnet18_C20_N65_mlp3_head4096/checkpoint_475.pth.tar --out ./saved_models/imagenet30/ours_flexmatch/resnet18_C20_N65_mlp3_head4096_ova_ckpt475_clf_threshold_ovaTh001_seed0/ --arch resnet_imagenet --lambda_x 1 --lambda_u 1 --lambda_oem 0.1 --lambda_socr 0.5 --batch-size 64 --lr 0.03 --skip_expand --expand-labels --seed 0 --opt_level O2 --mu 2 --no-progress --both_head_increment mlp3 --hidden_dim_increment 4096 --return_idx --ova_neg_loss all --ova_neg_DA ws --ova_unlabeled_threshold 0.01 --ova_unlabeled_neg_DA s --threshold 0.95