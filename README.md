# README 
This is a README file for our MLHC project: Advancing Thyroid Nodule Segmentation in Ultrasound Imaging: Leveraging SAM and Frequency Augmentation.
The purpose of this work is to improve the segmentation of ultrasound scans of the thyroid gland. As a part of our research, we set two goals: (1) Obtaining better IoU and DSC scores and (2) using fewer training examples.

**The project include this main folders and files:**
 - folder 1_or_data: 								              DDTI datasets
 - folder 2_preprocessed_data: 					          Preprocessing of DDTI dataset (remove duplications and crop)
 - AutoSAM:										                    The official repository for AutoSAM
 - MedSAM: 										                    The official repository for MedSAM
 - step2_TrainAndValidate.py:						          The main file for training and validatation					 
 - step3_TestOrInference.py:						          In this file we perform inference on original unprocessed images
 - step3_TestOrInference_medsam_boundingbox.py:	  In this file we perform inference on medSAM with bounding box
 - step3_TestOrInference_sam_boundingbox.py:      In this file we perform inference on SAM with bounding box
 - step3_TestOrInference_sam_gnenral.py           In this file we perform inference on SAM with his own generator
 - step3_TestOrInference_sam_mask.py              In this file we perform inference on SAM with his own generator
 - step3_TestOrInference_autosam.py               In this file we perform inference on autoSAM (after finetune)


# How to run this code?
Here, we split the whole process into **3 steps** so that you can easily replicate our results or perform the whole pipeline on your private custom dataset. 
 - step1, preparation of environment
 - step2, run the script [`step2_TrainAndValidate.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step2_TrainAndValidate.py) to train and validate the CNN model
 - step3, run the script [`step3_TestOrInference.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step3_TestOrInference.py) to testing model on original unprocessed images

## Step1 preparation of environment
We have tested our code in following environment：
 - [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) == 0.1.0 (with install command `pip install segmentation-models-pytorch`)
 - [`ttach`](https://github.com/qubvel/ttach) == 0.0.3
 - `torch` >＝1.0.0
 - `torchvision`
 - [`imgaug`](https://github.com/aleju/imgaug)

For installing `segmentation_models_pytorch (smp)`, there are two commands：
```linux
pip install segmentation-models-pytorch
```
Use the above command, you can install the released version that can run with `torch` <= 1.1.0, but without some decoders such as deeplabV3, deeplabV3+ and PAN.

```linux
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
Use the above command, you will install the latest version which includes decoders such as deeplabV3, deeplabV3+ and PAN, but requiring `torch` >= 1.2.0.


## Step2 training and validatation
In step2, you should run the script 
[`step2_TrainAndValidate.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step2_TrainAndValidate.py)
to train the first or second network in the cascaded framework (you need to train the two networks separately), **that is, the training process (such as hyperparameters) is the same for both first and second network except for the different training datasets used.** With the csv file `train.csv`, you can directly perform K-fold cross validation (default is 5-fold), and the script uses a fixed random seed to ensure that the K-fold cv of each experiment is repeatable (please note that the same random seed may cause different K-fold under different versions of `scikit-learn`).

**Our dataset include (2_preprocessed_data):**
 - two folders store PNG format images and masks respectively. Note that the file names (ID) of the images and masks should be integers and correspond to each other
 - a csv file that stores the image names (ID) and category. If there is only a segmentation task without a classification task, you can just falsify the category. For example, half of the data randomly is positive , and the other half is negative

Here is an example of how we trained the first and the second stages: 
stage 1: 
```python
python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage1_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage1/p_image --filepath_mask=./2_preprocessed_data/stage1/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name densenet201 --encoder_weights imagenet --train_limit 200
```

stage 2:
```python
python3 step2_TrainAndValidate.py --Task_name=dpv3plus_stage2_ --csv_file=./2_preprocessed_data/train.csv --filepath_img=./2_preprocessed_data/stage2/p_image --filepath_mask=./2_preprocessed_data/stage2/p_mask --fold_idx=1 --cuda_idx 0 --aug_type difficult --num_epochs 401 --num_epochs_decay 60 --decay_step 60 --batch_size 20 --num_workers 8 --encoder_name densenet201 --encoder_weights imagenet --train_limit 200`
```

Where:
```python
csv_file - the CSV file mentioned in the dataset 
filepath_img - The path to the images dataset
filepath_mask - The path to the masks dataset
fold_idx - Complete the training of all K folds
cuda_idx - The index of the GPU we want to use  
aug_type - The augmentation level we want to perform easy or difficult
num_epochs - The number of epochs 
num_epochs_decay - The minimum number of epochs that decay starts
decay_step - The amount of epochs we do between updates of the learning rate
batch_size - The size of each batch loaded at each step of training 
num_workers - The number of parallel processes we want to run in order to accelerate execution
train_limit - How many samples from training set do we take
encoder_weights - Pre-trained weights to load to model
encoder_name - Select encoder model type 
```

## Step3 testing (or inference or predicting)
In step3, you should run the script 
[`step3_TestOrInference.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step3_TestOrInference.py)
to perform inference based on the original unprocessed image.
Note: The location of the 2 trained models should be specified in the file with the parameter **selected**- the trained networks.

# In addition, for each type of medSAM or autoSAM model we created a different file as it mentioned above-
To perform inference on medSAM with his own generator run:
```python
python3 step3_TestOrInference_sam_gnenral.py
```

To perform inference on SAM with his own generator run:
```python
python3 step3_TestOrInference_sam_gnenral.py
```

To perform inference on SAM with bounding box run:
```python
python3 step3_TestOrInference_sam_boundingbox.py
```

To perform inference on medSAM with bounding box run:
```python
python3 step3_TestOrInference_medsam_boundingbox.py
```

To perform fine-tune of autoSAM run:
```python
python3 AutoSAM/scripts/main_autosam_seg.py
```

To perform inference on autoSAM (after finetune) run:
```python
python3 step3_TestOrInference_autosam.py
```

Here are some parameters should be noticed (Especially `c1_size` and `c2_size`):
```python
fold_flag = False # False for inference on new data, and True for cross-validation
mask_path = None  # None for inference on new data
save_path = None  # If you don't want to save the inference result, set it to None
c1_size = 256  # The input image size for training the first network ❗
c1_tta = False # Whether to use TTA for the first network
orimg = False    # False for no preprocessing to remove irrelevant areas
use_c2_flag = False # Whether to use the second network
c2_size = 512 # The input image size for training the second network ❗
c2_tta = False # Whether to use TTA for the second network
autosam_path = r'./AutoSAM/scripts/model_best_sample.pth.tar' # The path of the trained model of AutoSAM
medsam_path = './MedSAM/work_dir/MedSAM/medsam_vit_b.pth' # The path of the trained model of MedSAM
```
