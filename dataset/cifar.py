import logging
import math
import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
from .mydataset import ImageFolder, ImageFolder_fix

logger = logging.getLogger(__name__)

__all__ = ['TransformOpenMatch', 'TransformOpenMatch_strong', 'TransformFixMatch', 'cifar10_mean',
           'cifar10_std', 'cifar100_mean', 'cifar100_std', 'normal_mean',
           'normal_std', 'TransformFixMatch_Imagenet', 'TransformFixMatch_2strong',
           'TransformFixMatch_Imagenet_Weak', 'TransformFixMatch_Imagenet_2strong']
# Enter Path of the data directory.
DATA_PATH = './data'

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar(args, norm=True):
    root = args.root
    name = args.dataset
    if name == "cifar10":
        data_folder = datasets.CIFAR10
        data_folder_main = CIFAR10SSL
        mean = cifar10_mean
        std = cifar10_std
        num_class = 10
    elif name == "cifar100":
        data_folder = CIFAR100FIX
        data_folder_main = CIFAR100SSL
        mean = cifar100_mean
        std = cifar100_std
        num_class = 100
        num_super = args.num_super

    else:
        raise NotImplementedError()
    assert num_class >= args.num_classes

    if name == "cifar10":
        base_dataset = data_folder(root, train=True, download=True)
        args.num_classes = 6
    elif name == 'cifar100':
        base_dataset = data_folder(root, train=True,
                                   download=True, num_super=num_super)
        args.num_classes = base_dataset.num_known_class

    base_dataset.targets = np.array(base_dataset.targets)
    if name == 'cifar10':
        base_dataset.targets -= 2
        base_dataset.targets[np.where(base_dataset.targets == -2)[0]] = 8
        base_dataset.targets[np.where(base_dataset.targets == -1)[0]] = 9

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = \
        x_u_split(args, base_dataset.targets)

    ## This function will be overwritten in trainer.py
    norm_func = TransformFixMatch(mean=mean, std=std, norm=norm)
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    if name == 'cifar10':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, train=True, return_idx=args.return_idx,
            transform=norm_func)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, train=True,
            transform=norm_func, return_idx=args.return_idx)
        val_dataset = data_folder_main(
            root, val_idxs, train=True,
            transform=norm_func_test)
    elif name == 'cifar100':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, num_super = num_super, train=True, return_idx=args.return_idx,
            transform=norm_func)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, num_super = num_super, train=True,
            transform=norm_func, return_idx=args.return_idx)
        val_dataset = data_folder_main(
            root, val_idxs, num_super = num_super,train=True,
            transform=norm_func_test)

    if name == 'cifar10':
        train_labeled_dataset.targets -= 2
        train_unlabeled_dataset.targets -= 2
        val_dataset.targets -= 2
        train_unlabeled_dataset.targets[np.where(train_unlabeled_dataset.targets == -2)[0]] = 8
        train_unlabeled_dataset.targets[np.where(train_unlabeled_dataset.targets == -1)[0]] = 9
        target_ind = np.where(train_unlabeled_dataset.targets >= args.num_classes)[0]
        train_unlabeled_dataset.targets[target_ind] = args.num_classes

    if name == 'cifar10':
        test_dataset = data_folder(
            root, train=False, transform=norm_func_test, download=False)
    elif name == 'cifar100':
        test_dataset = data_folder(
            root, train=False, transform=norm_func_test,
            download=False, num_super=num_super)
    test_dataset.targets = np.array(test_dataset.targets)

    if name == 'cifar10':
        test_dataset.targets -= 2
        test_dataset.targets[np.where(test_dataset.targets == -2)[0]] = 8
        test_dataset.targets[np.where(test_dataset.targets == -1)[0]] = 9

    target_ind = np.where(test_dataset.targets >= args.num_classes)[0]
    test_dataset.targets[target_ind] = args.num_classes


    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)
    logger.info("Dataset: %s"%name)
    logger.info(f"Labeled examples: {len(unique_labeled)}"
                f"Unlabeled examples: {len(train_unlabeled_idxs)}"
                f"Valdation samples: {len(val_labeled)}")
    return train_labeled_dataset, train_unlabeled_dataset, \
           test_dataset, val_dataset


def get_cross(args, norm=True):
    root = args.root
    mean = cifar100_mean
    std = cifar100_std
    args.num_classes = 50

    base_dataset = datasets.CIFAR100(root, train=True, download=False)
    base_dataset.targets = np.array(base_dataset.targets)
    valid_idx = np.where(base_dataset.targets < args.num_classes)[0]
    base_dataset.targets = base_dataset.targets[valid_idx]

    def cross_x_u_split(args, labels):
        label_per_class = args.num_labeled  # // args.num_classes
        val_per_class = args.num_val  # // args.num_classes
        labels = np.array(labels)
        labeled_idx = []
        val_idx = []
        unlabeled_idx = []
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        for i in range(args.num_classes):
            idx = np.where(labels == i)[0]
            unlabeled_idx.extend(idx)
            idx = np.random.choice(idx, label_per_class + val_per_class, False)
            labeled_idx.extend(idx[:label_per_class])
            val_idx.extend(idx[label_per_class:])

        labeled_idx = np.array(labeled_idx)

        # if not args.no_out:
        unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
        unlabeled_idx = [idx for idx in unlabeled_idx if idx not in val_idx]
        assert len(labeled_idx) + len(unlabeled_idx) + len(
            val_idx) == args.num_classes * 500, f"{len(labeled_idx)} + {len(unlabeled_idx)} + {len(val_idx)} is not {args.num_classes * 500}"

        assert len(labeled_idx) == args.num_labeled * args.num_classes
        if args.skip_expand:
            pass
        else:
            if args.expand_labels or args.num_labeled * args.num_classes < args.batch_size:
                num_expand_x = math.ceil(
                    args.batch_size * args.eval_step / args.num_labeled / args.num_classes)
                labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)

        return labeled_idx, unlabeled_idx, val_idx

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = cross_x_u_split(args, base_dataset.targets)

    ## This function will be overwritten in trainer.py
    norm_func = TransformFixMatch(mean=mean, std=std, norm=norm)
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_labeled_dataset = CrossSSL(
        root, train_labeled_idxs, train=True, return_idx=args.return_idx, transform=norm_func)
    train_unlabeled_dataset = CrossSSL(
        root, train_unlabeled_idxs, train=True, transform=norm_func, return_idx=args.return_idx, is_ulb=True)
    val_dataset = CrossSSL(root, val_idxs, train=True, transform=norm_func_test)
    test_dataset = CrossSSL(root, None, train=False, transform=norm_func_test)

    unique, counts = np.unique(train_labeled_dataset.targets, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(train_unlabeled_dataset.targets, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(val_dataset.targets, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(test_dataset.targets, return_counts=True)
    print(dict(zip(unique, counts)))

    # Note that test_dataset actually should not contain the latter 50 classes, type-1 ood does not makes sense here
    # test_dataset = datasets.CIFAR100(root, train=False, transform=norm_func_test, download=False)
    # test_dataset.targets = np.array(test_dataset.targets)
    # test_dataset.targets[test_dataset.targets >= args.num_classes] = args.num_classes

    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)
    logger.info("Dataset: Cross")
    logger.info(f"Labeled examples: {len(unique_labeled)}"
                f"Unlabeled examples: {len(train_unlabeled_idxs)}"
                f"Valdation samples: {len(val_labeled)}")
    return train_labeled_dataset, train_unlabeled_dataset, \
           test_dataset, val_dataset


def get_imagenet(args, norm=True):
    mean = normal_mean
    std = normal_std
    if args.imgnet_percent == 10:
        txt_labeled = "./imagenet_filelist/imagenet_train_labeled.txt"
        txt_unlabeled = "./imagenet_filelist/imagenet_train_unlabeled.txt"
    elif args.imgnet_percent == 5:
        txt_labeled = "./imagenet_filelist/imagenet_train_labeled_5percent.txt"
        txt_unlabeled = "./imagenet_filelist/imagenet_train_unlabeled_5percent.txt"
    elif args.imgnet_percent == 1:
        txt_labeled = "./imagenet_filelist/imagenet_train_labeled_1percent.txt"
        txt_unlabeled = "./imagenet_filelist/imagenet_train_unlabeled_1percent.txt"
    else:
        raise NotImplementedError
    txt_val = "./imagenet_filelist/imagenet_val.txt"
    txt_test = "./imagenet_filelist/imagenet_test.txt"
    # This function will be overwritten in trainer.py
    norm_func = TransformFixMatch_Imagenet(mean=mean, std=std,
                                           norm=norm, size_image=224)
    dataset_labeled = ImageFolder(txt_labeled, transform=norm_func, return_idx=args.return_idx)
    dataset_unlabeled = ImageFolder_fix(txt_unlabeled, transform=norm_func, return_idx=args.return_idx)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset_val = ImageFolder(txt_val, transform=test_transform)
    dataset_test = ImageFolder(txt_test, transform=test_transform)
    logger.info(f"Labeled examples: {len(dataset_labeled)}"
                f"Unlabeled examples: {len(dataset_unlabeled)}"
                f"Valdation samples: {len(dataset_val)}")
    return dataset_labeled, dataset_unlabeled, dataset_test, dataset_val


def x_u_split(args, labels):
    label_per_class = args.num_labeled #// args.num_classes
    val_per_class = args.num_val #// args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx)
        idx = np.random.choice(idx, label_per_class+val_per_class, False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])

    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == args.num_labeled * args.num_classes
    if args.skip_expand:
        pass
    else:
        if args.expand_labels or args.num_labeled * args.num_classes < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.eval_step / args.num_labeled / args.num_classes)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    #if not args.no_out:
    unlabeled_idx = np.array(range(len(labels)))
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in val_idx]
    return labeled_idx, unlabeled_idx, val_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong


class TransformCCSSL(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])

        self.simclr = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.simclr(x)
        else:
            return weak, strong


class TransformCCSSL_2strong(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.simclr = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.strong(x)), self.normalize(self.weak2(x)), self.simclr(x)
        else:
            return weak, strong


class TransformFixMatch_2strong(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.strong(x)), self.normalize(self.weak2(x))
        else:
            return weak, strong


class TransformOpenMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong


class TransformOpenMatch_strong(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.strong(x))
        else:
            return weak, strong


class TransformFixMatch_Imagenet(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong


class TransformFixMatch_Imagenet_Weak(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak2(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong


class TransformFixMatch_Imagenet_2strong(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak2(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.strong(x)), self.normalize(weak2)
        else:
            return weak, strong


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class CIFAR100FIX(datasets.CIFAR100):
    def __init__(self, root, num_super=10, train=True, transform=None,
                 target_transform=None, download=False, return_idx=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.course_labels = coarse_labels[self.targets]
        self.targets = np.array(self.targets)
        labels_unknown = self.targets[np.where(self.course_labels > num_super)[0]]
        labels_known = self.targets[np.where(self.course_labels <= num_super)[0]]
        unknown_categories = np.unique(labels_unknown)
        known_categories = np.unique(labels_known)

        num_unknown = len(unknown_categories)
        num_known = len(known_categories)
        print("number of unknown categories %s"%num_unknown)
        print("number of known categories %s"%num_known)
        assert num_known + num_unknown == 100
        #new_category_labels = list(range(num_known))
        self.targets_new = np.zeros_like(self.targets)
        for i, known in enumerate(known_categories):
            ind_known = np.where(self.targets==known)[0]
            self.targets_new[ind_known] = i
        for i, unknown in enumerate(unknown_categories):
            ind_unknown = np.where(self.targets == unknown)[0]
            self.targets_new[ind_unknown] = num_known

        self.targets = self.targets_new
        assert len(np.where(self.targets == num_known)[0]) == len(labels_unknown)
        assert len(np.where(self.targets < num_known)[0]) == len(labels_known)
        self.num_known_class = num_known


    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(CIFAR100FIX):
    def __init__(self, root, indexs, num_super=10, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, num_super=num_super,train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idx = return_idx
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        self.set_index()
    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets


    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class CrossSSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False, is_ulb=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.num_known_class = 50
        self.targets = np.array(self.targets)
        valid_idx = np.where(self.targets < self.num_known_class)[0]
        self.data = self.data[valid_idx]
        self.targets = self.targets[valid_idx]
        self.return_idx = return_idx

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        assert all(self.targets < self.num_known_class)
        if not train:
            tin_data = np.load('/BS/yfan/nobackup/tiny-imagenet/TIN_test_resize.npy')
            tin_targets = np.ones((tin_data.shape[0],), dtype=int) * self.num_known_class
            self.data = np.concatenate([self.data, tin_data], axis=0)
            self.targets = np.concatenate([self.targets, tin_targets], axis=0)
        else:
            if is_ulb:
                tin_data = np.load('/BS/yfan/nobackup/tiny-imagenet/TIN_train_resize.npy')
                tin_targets = np.ones((tin_data.shape[0], ), dtype=int) * self.num_known_class
                self.data = np.concatenate([self.data, tin_data], axis=0)
                self.targets = np.concatenate([self.targets, tin_targets], axis=0)
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class CIFAR_cocnat(torch.utils.data.Dataset):

    def __init__(self, labeled_data, labeled_targets, unlabeled_data, unlabeled_targets,
                 start_label=0, transform=None, target_transform=None):
        super(CIFAR_cocnat, self).__init__()

        self.transform = transform
        self.target_transform = target_transform

        self.data_x = labeled_data
        self.targets_x = np.array(labeled_targets)

        self.data_u = unlabeled_data
        self.targets_u = np.array(unlabeled_targets)

        self.soft_labels = np.zeros((len(self.data_x) + len(self.data_u)), dtype=np.float32)
        for idx in range(len(self.data_x) + len(self.data_u)):
            if idx < len(self.data_x):
                self.soft_labels[idx] = 1.0
            else:
                self.soft_labels[idx] = start_label
        self.prediction = np.zeros((len(self.data_x) + len(self.data_u), 10), dtype=np.float32)
        self.prediction[:len(self.data_x), :] = 1.0
        self.count = 0

    def __len__(self):
        return len(self.data_x) + len(self.data_u)

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[len(self.data_x):, idx] = results[len(self.data_x):]

        if self.count >= 10:
            self.soft_labels = self.prediction.mean(axis=1)

    def __getitem__(self, index):
        if index < len(self.data_x):
            img, target = self.data_x[index], self.targets_x[index]
        else:
            img, target = self.data_u[index - len(self.data_x)], self.targets_u[index - len(self.data_x)]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.soft_labels[index], index


def get_transform(mean, std, image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_ood(dataset, id, test_only=False, image_size=None):
    image_size = (32, 32, 3) if image_size is None else image_size
    if id == "cifar10":
        mean = cifar10_mean
        std = cifar10_std
    elif id == "cifar100":
        mean = cifar100_mean
        std = cifar100_std
    elif "imagenet"  in id or id == "tiny":
        mean = normal_mean
        std = normal_std
    elif id == 'cross':
        mean = cifar100_mean
        std = cifar100_std

    _, test_transform = get_transform(mean, std, image_size=image_size)

    if dataset == 'cifar10':
        test_dir = os.path.join(DATA_PATH, 'cifar-10')
        test_set = datasets.CIFAR10(test_dir, train=False, download=False,
                                    transform=test_transform)

    elif dataset == 'cifar100':
        test_dir = os.path.join(DATA_PATH, 'cifar-100')
        test_set = datasets.CIFAR100(test_dir, train=False, download=False,
                                     transform=test_transform)

    elif dataset == 'svhn':
        test_dir = os.path.join(DATA_PATH, 'svhn')
        test_set = datasets.SVHN(test_dir, split='test', download=True,
                                 transform=test_transform)

    elif dataset == 'lsun':
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'stanford_dogs':
        test_dir = os.path.join(DATA_PATH, 'Stanford_Dogs', 'Images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'cub':
        test_dir = os.path.join(DATA_PATH, 'CUB_200_2011', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'flowers102':
        test_dir = os.path.join(DATA_PATH, 'oxford_flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'food_101':
        test_dir = os.path.join(DATA_PATH, 'food-101-no-hotdog', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'caltech_256':
        test_dir = os.path.join(DATA_PATH, '256_ObjectCategories')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'dtd':
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'pets':
        test_dir = os.path.join(DATA_PATH, 'pet_for_openmatch')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    return test_set


DATASET_GETTERS = {'cifar10': get_cifar,
                   'cifar100': get_cifar,
                   'imagenet': get_imagenet,
                   'cross': get_cross,
                   }
