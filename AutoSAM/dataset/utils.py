import os
import pickle
from AutoSAM.dataset.Synapse import SynapseDataset
from AutoSAM.dataset.ACDC import AcdcDataset
# from AutoSAM.dataset.SliceLoader import SliceDataset
import torch
from tnscui_utils.TNSUCI_util import *
from loader.data_loader import get_loader, get_loader_difficult


def generate_dataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']
    test_keys = splits[args.fold]['test']

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]

    print(tr_keys)
    print(val_keys)
    print(test_keys)


    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        train_ds = AcdcDataset(keys=tr_keys, mode='train', args=args)
        val_ds = AcdcDataset(keys=val_keys, mode='val', args=args)
        test_ds = AcdcDataset(keys=test_keys, mode='val', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    )

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler

def generate_dataset_miccai(args):
    if args.validate_flag:
        train, valid, test = get_fold_filelist(args.csv_file, K=args.fold_K, fold=args.fold_idx, validation=True)
    else:
        train, test = get_fold_filelist(args.csv_file, K=args.fold_K, fold=args.fold_idx)

    train_list = [args.filepath_img + sep + i[0] for i in train][:args.num_samples_train]
    train_list_GT = [args.filepath_mask + sep + i[0] for i in train][:args.num_samples_train]

    test_list = [args.filepath_img+sep+i[0] for i in test][:args.num_samples_test]
    test_list_GT = [args.filepath_mask+sep+i[0] for i in test][:args.num_samples_test]

    if args.validate_flag:
        valid_list = [args.filepath_img+sep+i[0] for i in valid][:args.num_samples_val]
        valid_list_GT = [args.filepath_mask+sep+i[0] for i in valid][:args.num_samples_val]
    else:
        # just copy test as validation,
        # also u can get the real valid_list use the func 'get_fold_filelist' by setting the param 'validation' as True
        valid_list = test_list[:args.num_samples_val]
        valid_list_GT = test_list_GT[:args.num_samples_val]

    if args.aug_type == 'easy':
        print('augmentation with easy level')
        train_loader = get_loader(seg_list=None,
                                  GT_list = train_list_GT,
                                  image_list=train_list,
                                  image_size=args.img_size,
                                  batch_size=args.batch_size,
                                  num_workers=args.workers,
                                  mode='train',
                                  augmentation_prob=args.augmentation_prob,)
    elif args.aug_type == 'difficult':
        print('augmentation with difficult level')
        train_loader = get_loader_difficult(seg_list=None,
                                              GT_list=train_list_GT,
                                              image_list=train_list,
                                              image_size=args.img_size,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              mode='train',
                                              augmentation_prob=args.augmentation_prob,)
    else:
        raise('difficult or easy')
    valid_loader = get_loader(seg_list=None,
                              GT_list = valid_list_GT,
                            image_list=valid_list,
                            image_size=args.img_size,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            mode='valid',
                            augmentation_prob=0.,)

    test_loader = get_loader(seg_list=None,
                             GT_list = test_list_GT,
                            image_list=test_list,
                            image_size=args.img_size,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            mode='test',
                            augmentation_prob=0.,)

    # Not supporting distributed if it is important want to fix see generate_dataset
    train_sampler = None
    val_sampler = None
    test_sampler = None

    return train_loader, train_sampler, valid_loader, val_sampler, test_loader, test_sampler


def generate_test_loader(key, args):
    key = [key]
    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        test_ds = AcdcDataset(keys=key, mode='val', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    )

    return test_loader


def generate_contrast_dataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]

    print(tr_keys)
    print(val_keys)

    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        train_ds = AcdcDataset(keys=tr_keys, mode='contrast', args=args)
        val_ds = AcdcDataset(keys=val_keys, mode='contrast', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    return train_loader, val_loader
