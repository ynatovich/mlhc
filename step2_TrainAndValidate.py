import argparse

from torch.backends import cudnn

from loader.data_loader import get_loader, get_loader_difficult
from tnscui_utils.TNSUCI_util import *
from tnscui_utils.solver import Solver as Solver_or
from datetime import datetime

def main(config):
    cudnn.benchmark = True
    now = datetime.now()
    dt_string = now.strftime("_Time_%d_%m_%Y__%H_%M_%S")
    config.result_path = os.path.join(config.result_path, config.Task_name+str(config.fold_K)+'_'+str(config.fold_idx)+'_'+str(config.encoder_name)+dt_string)
    print(config.result_path)
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        os.makedirs(config.model_path)
        os.makedirs(config.log_dir)
        os.makedirs(os.path.join(config.result_path,'images'))


    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)


    print(config)
    f = open(os.path.join(config.result_path,'config.txt'),'w')
    for key in config.__dict__:
        print('%s: %s'%(key, config.__getattribute__(key)), file=f)
    f.close()

    if config.validate_flag:
        train, valid, test = get_fold_filelist(config.csv_file, K=config.fold_K, fold=config.fold_idx, validation=True)
    else:
        train, test = get_fold_filelist(config.csv_file, K=config.fold_K, fold=config.fold_idx)
    """
    if u want to use fixed folder as img & mask folder, u can use following code 
    
    train_list = get_filelist_frompath(train_img_folder,'PNG') 
    train_list_GT = [train_mask_folder+sep+i.split(sep)[-1] for i in train_list]
    test_list = get_filelist_frompath(test_img_folder,'PNG') 
    test_list_GT = [test_mask_folder+sep+i.split(sep)[-1] for i in test_list]
    
    """

    trainMax = np.min((config.train_limit,len(train)))
    train_list = [config.filepath_img+sep+i[0] for i in train[0:trainMax]]
    train_list_GT = [config.filepath_mask+sep+i[0] for i in train[0:trainMax]]

    test_list = [config.filepath_img+sep+i[0] for i in test]
    test_list_GT = [config.filepath_mask+sep+i[0] for i in test]

    if config.validate_flag:
        valid_list = [config.filepath_img+sep+i[0] for i in valid]
        valid_list_GT = [config.filepath_mask+sep+i[0] for i in valid]
    else:
        # just copy test as validation,
        # also u can get the real valid_list use the func 'get_fold_filelist' by setting the param 'validation' as True
        valid_list = test_list
        valid_list_GT = test_list_GT



    config.train_list = train_list
    config.test_list = test_list
    config.valid_list = valid_list

    if config.aug_type == 'easy':
        print('augmentation with easy level')
        train_loader = get_loader(seg_list=None,
                                  GT_list = train_list_GT,
                                  image_list=train_list,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob,)
    elif config.aug_type == 'difficult':
        print('augmentation with difficult level')
        train_loader = get_loader_difficult(seg_list=None,
                                              GT_list=train_list_GT,
                                              image_list=train_list,
                                              image_size=config.image_size,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers,
                                              mode='train',
                                              augmentation_prob=config.augmentation_prob,)
    else:
        raise('difficult or easy')
    valid_loader = get_loader(seg_list=None,
                              GT_list = valid_list_GT,
                            image_list=valid_list,
                            image_size=config.image_size,
                            batch_size=config.batch_size_test,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.,)

    test_loader = get_loader(seg_list=None,
                             GT_list = test_list_GT,
                            image_list=test_list,
                            image_size=config.image_size,
                            batch_size=config.batch_size_test,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.,)

    '''(['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    	  'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107',
    	  'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154',
    	  'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121',
    	  'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0',
    	  'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
    	  'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception'])
    	  '''
    encoder_name = config.encoder_name
    encoder_weights = config.encoder_weights
    solver = Solver_or(config, train_loader, valid_loader, test_loader, encoder_name, encoder_weights)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        unet_path = os.path.join(config.model_path, 'best_unet_score.pkl')
        if config.tta_mode:
            print(char_color('@,,@   doing with tta test'))
            acc, SE, SP, PC, DC, IOU = solver.test_tta(mode='test', unet_path=unet_path)
        else:
            acc, SE, SP, PC, DC, IOU = solver.test(mode='test', unet_path = unet_path)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
            acc, SE, SP, PC, DC, IOU))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)  #The size of the network input img, that is, the input will be forced to resize to this size

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=405)

    parser.add_argument('--num_epochs_decay', type=int, default=60)  # The minimum number of epochs that decay starts
    parser.add_argument('--decay_ratio', type=float, default=0.01) # 0~1, each decay to 1*ratio
    parser.add_argument('--decay_step', type=int, default=60)  # epoch

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--batch_size_test', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=3)

    # Set Learning Rate
    parser.add_argument('--lr', type=float, default=1e-4)  # Initial or maximum learning rate (when using lovz alone and multiple GPUs, lr seems to be larger to converge)
    parser.add_argument('--lr_low', type=float, default=1e-12)  # The minimum learning rate, if it is set to None, it will be 1e+6 of the maximum learning rate (cannot be set to 0)

    parser.add_argument('--lr_warm_epoch', type=int, default=5)  # The epoch number of warmup is generally 5~20, if it is 0 or False, it will not be used
    parser.add_argument('--lr_cos_epoch', type=int, default=350)  # The epoch number of cos annealing is generally the total epoch number-warmup number, if it is 0 or False, it means not used

    # optimizer param
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam

    parser.add_argument('--augmentation_prob', type=float, default=1.0)  # Amplification probability

    parser.add_argument('--save_model_step', type=int, default=20)
    parser.add_argument('--val_step', type=int, default=1)


    # misc
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--tta_mode', type=bool, default=True) # Whether to use tta for validation during training
    parser.add_argument('--Task_name', type=str, default='test', help='DIR name,Task name')
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--DataParallel', type=bool, default=False) ##

    # data-parameters
    parser.add_argument('--filepath_img', type=str, default='./1_or_data/image')
    parser.add_argument('--filepath_mask', type=str, default='./1_or_data/mask')
    parser.add_argument('--csv_file', type=str, default='./2_preprocessed_data/train.csv')
    parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')
    parser.add_argument('--fold_idx', type=int, default=1)

    # result&save
    parser.add_argument('--result_path', type=str, default='./result/TNSCUI')
    parser.add_argument('--save_detail_result', type=bool, default=True)
    parser.add_argument('--save_image', type=bool, default=True) # Observe images and results during training

    # more param
    parser.add_argument('--test_flag', type=bool, default=False) # Whether to test during training, not testing will save a lot of time
    parser.add_argument('--validate_flag', type=bool, default=False) # Is there a validation set
    parser.add_argument('--aug_type', type=str, default='difficult', help='difficult or easy') # Amplify the code during the training process, divided into dasheng, shaonan
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b7', help='encoder type') # Select encoder model type (['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception'])
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='encoder type') # Select encoder init weights depending on model type  can be 'None', 'imagenet', 'instagram' or 'url'
    parser.add_argument('--train_limit', type=int, default=10000) # Select maximal number of train samples to test few shot

    config = parser.parse_args()
    main(config)



