#!/bin/bash

#############################################################################################################
# 1
python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 8 --encoder_name efficientnet-b4 --encoder_weights imagenet

python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 8 --encoder_name efficientnet-b4 --encoder_weights imagenet

#############################################################################################################
# 2
python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name efficientnet-b5

python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name efficientnet-b5

#############################################################################################################
# 3
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 8 --encoder_name resnet50 --encoder_weights imagenet

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 8 --encoder_name resnet50 --encoder_weights imagenet

#############################################################################################################
# 4
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name resnet101 --encoder_weights imagenet

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name resnet101 --encoder_weights imagenet

#############################################################################################################
# 5
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 16 --num_workers 8 --encoder_name resnet152 --encoder_weights imagenet

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 16 --num_workers 8 --encoder_name resnet152 --encoder_weights imagenet

#############################################################################################################
# 6
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 20 --encoder_name resnext50_32x4d --encoder_weights imagenet

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 20 --encoder_name resnext50_32x4d --encoder_weights imagenet

#############################################################################################################
# 7 init instagram
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name resnext101_32x16d --encoder_weights instagram

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name resnext101_32x16d --encoder_weights instagram

#############################################################################################################
# 8 init instagram
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 6 --num_workers 8 --encoder_name resnext101_32x48d --encoder_weights instagram

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 6 --num_workers 8 --encoder_name resnext101_32x48d --encoder_weights instagram

#############################################################################################################
# 9 init imagenet
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 8 --encoder_name se_resnet50 --encoder_weights imagenet

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 40 --num_workers 8 --encoder_name se_resnet50 --encoder_weights imagenet

#############################################################################################################
# 10 init imagenet
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name se_resnet101 --encoder_weights imagenet

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name se_resnet101 --encoder_weights imagenet


#############################################################################################################
# 11 init weights url
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name densenet201 --encoder_weights imagenet

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name densenet201 --encoder_weights imagenet


#############################################################################################################
# 12 init weights url
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name inceptionresnetv2 --encoder_weights url

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name inceptionresnetv2 --encoder_weights url


#############################################################################################################
# 13 init weights url
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name inceptionv4 --encoder_weights url

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name inceptionv4 --encoder_weights url


#############################################################################################################
# 14 init weights url
#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name xception --encoder_weights url

#python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name xception --encoder_weights url

#############################################################################################################
# 15
python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 10 --num_workers 8 --encoder_name efficientnet-b6

python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 10 --num_workers 8 --encoder_name efficientnet-b6

#############################################################################################################
# 16
python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name efficientnet-b7

python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name efficientnet-b7

