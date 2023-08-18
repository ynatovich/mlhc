# Calculate dice one by one, and then save
# Calculating dice is not the ultimate goal, this script should be used as the final image output script
import torch
import ttach as tta
from PIL import Image
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.transform import resize
from torchvision import transforms as T

import segmentation_models_pytorch_4TorchLessThan120 as smp
from tnscui_utils.TNSCUI_preprocess import TNSCUI_preprocess
from tnscui_utils.TNSUCI_util import *
import matplotlib.pyplot as plt

def getIOU(SR,GT):
    """
    are binary images
    :param SR: 
    :param GT: 
    :return: 
    """
    TP = (SR+GT==2).astype(np.float32)
    FP = (SR+(1-GT)==2).astype(np.float32)
    FN = ((1-SR)+GT==2).astype(np.float32)

    IOU = float(np.sum(TP))/(float(np.sum(TP+FP+FN)) + 1e-6)

    return IOU

def getDSC(SR,GT):
    """
    are binary images
    :param SR: 
    :param GT: 
    :return: 
    """


    Inter = np.sum(((SR+GT)==2).astype(np.float32))
    DC = float(2*Inter)/(float(np.sum(SR)+np.sum(GT)) + 1e-6)

    return DC

def largestConnectComponent(bw_img):
    if np.sum(bw_img)==0:
        return bw_img
    #labeled_img, num = sklabel(bw_img, neighbors=4, background=0, return_num=True)
    labeled_img, num = sklabel(bw_img, background=0, connectivity=1, return_num=True)
    if num == 1:
        return bw_img


    max_label = 0
    max_num = 0
    for i in range(0,num):
        print(i)
        if np.sum(labeled_img == (i+1)) > max_num:
            max_num = np.sum(labeled_img == (i+1))
            max_label = i+1
    mcr = (labeled_img == max_label)
    return mcr.astype(np.int)


def preprocess(mask_c1_array_biggest, c1_size=256):
    if np.sum(mask_c1_array_biggest) == 0:
            minr, minc, maxr, maxc = [0, 0, c1_size, c1_size]
    else:
        region = regionprops(mask_c1_array_biggest)[0]
        minr, minc, maxr, maxc = region.bbox

    dim1_center, dim2_center = [(maxr + minr) // 2, (maxc + minc) // 2]
    max_length = max(maxr - minr, maxc - minc)

    max_lengthl = int((c1_size/256)*80)
    preprocess1 = int((c1_size/256)*19)
    pp22 = int((c1_size/256)*31)


    if max_length > max_lengthl:
        ex_pixel = preprocess1 + max_length // 2
    else:
        ex_pixel = pp22 + max_length // 2

    dim1_cut_min = dim1_center - ex_pixel
    dim1_cut_max = dim1_center + ex_pixel
    dim2_cut_min = dim2_center - ex_pixel
    dim2_cut_max = dim2_center + ex_pixel

    if dim1_cut_min < 0:
        dim1_cut_min = 0
    if dim2_cut_min < 0:
        dim2_cut_min = 0
    if dim1_cut_max > c1_size:
        dim1_cut_max = c1_size
    if dim2_cut_max > c1_size:
        dim2_cut_max = c1_size
    return [dim1_cut_min,dim1_cut_max,dim2_cut_min,dim2_cut_max]


if __name__ == '__main__':
    """processing logic:
    1)Read the picture, read the mask if there is a mask
    2)Image preprocessing, such as the operation of removing black borders, retain the coordinate index1 of removing black borders
    3)cascade1, use TTA to predict mask (subsequent post-processing can be added, such as cascadePSP)
    4)Predict the roi of the mask, randomly expand different pixels up, down, left, and right, which is equivalent to a kind of tta, and then store the details of the expanded pixels of tta into a list, which is index2
    5)cascade2, use TTA, predict based on the cutout image (resize to 256), and then restore the size, (subsequent post-processing can be added, such as cascadePSP)
    6)According to index2, restore the segmentation result to the original position, and then restore to the original position according to index1
    7)and mask calculation indicators
    9)Save the predicted image to the specified folder
    """
    saveas = 'mask' # Save in the form of prob probability graph, save in the form of mask binary graph
    # Path setting
    img_path = r'./1_or_data/image'
    mask_path = r'./1_or_data/mask'
    csv_file = r'./2_preprocessed_data/train.csv'
    save_path = None

    # Whether to use the set split information
    fold_flag = True  # The prediction test set is set to False to directly read the PNG file in img_path for testing, and True to use the split information
    fold_K = 5
    fold_index = 1

    # task_name
    task_name = r' '

    # Main selected trainings
    # Unmark the selected line selecting the desired trained networks
    #selected = "Orig b6"       #dpv3plus_stage1_5_1_Time_19_08_2023__23_48_04
    #selected = "Freq b6"       #dpv3plus_stage1_5_1_Time_20_08_2023__23_49_27   ### Selected ####
    #selected = "Freq ressnext"  #dpv3plus_stage1_5_1_resnext101_32x16d_Time_09_09_2023__00_54_46
    ################################################################################
    #selected = "efficientnet b4"
    #selected = "efficientnet b4 2" # missing stage 1
    #selected = "efficientnet b5"
    #selected = "efficientnet b5 2"
    #selected = "efficientnet b6"   #### Selected Ref ######
    #selected = "efficientnet b7"
    #selected = "efficientnet b47"
    #selected = "efficientnet b57"
    #selected = "resnet50 efficientnet b7"
    #selected = "efficientnet b5 s2"
    #selected = "resnet50"
    #selected = "resnet101"
    #selected = "resnet152"
    #selected = "resnext50_32x4d"
    #selected = "resnext101_32x16d"
    #selected = "resnext101_32x48d"
##############################################
    #selected = "20S"
    selected = "20F"
    #selected = "40S"
    #selected = "40F"
    #selected = "100S"
    #selected = "100F"
    #selected = "200S"
    #selected = "200F"

    encoder_name2 = "Null"
    if selected == "Orig b6":
        # Train after BPF bs=20
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_Time_19_08_2023__23_48_04/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_Time_20_08_2023__08_27_45/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b6" #0.732 0.826
    elif selected == "Freq b6":
        # Train after BPF + LPF + HPF + original bs=20
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_Time_20_08_2023__23_49_27/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_Time_21_08_2023__07_47_51/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b6" #0.75 0.842
    elif selected == "Freq ressnext":
        # resnext101_32x16d
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnext101_32x16d_Time_09_09_2023__00_54_46/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_resnext101_32x16d_Time_09_09_2023__04_35_48/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnext101_32x16d"
    elif selected == "efficientnet b4":
        # efficientnet b4
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b4_Time_29_08_2023__20_36_25/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b4_Time_15_09_2023__17_55_27/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b4"
    elif selected == "efficientnet b4 2":
        # efficientnet b4 2
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b4_Time_29_08_2023__20_36_25/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b4_Time_18_09_2023__11_54_42/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b4"
    elif selected == "efficientnet b5":
        # efficientnet b5
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_Time_30_08_2023__03_22_19/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_Time_15_09_2023__21_19_19/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "efficientnet b5 2":
        # efficientnet b5 2
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_Time_18_09_2023__15_19_40/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_Time_18_09_2023__18_07_34/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "efficientnet b6":
        # efficientnet b6
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b6_Time_17_09_2023__10_55_14/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b6_Time_17_09_2023__14_27_32/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b6"
    elif selected == "efficientnet b7":
        # efficientnet b7
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b7_Time_17_09_2023__19_51_14/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b7_Time_17_09_2023__23_34_19/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b7"
    elif selected == "efficientnet b47":
        # efficientnet b4 b7
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b4_Time_29_08_2023__20_36_25/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b7_Time_17_09_2023__23_34_19/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b4"
        encoder_name2 = "efficientnet-b7"
    elif selected == "efficientnet b57":
        # efficientnet b5 b7
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_Time_30_08_2023__03_22_19/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b7_Time_17_09_2023__23_34_19/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
        encoder_name2 = "efficientnet-b7"
    elif selected == "resnet50 efficientnet b7":
        # resnet50 efficientnet b7
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnet50_Time_30_08_2023__10_37_43/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b7_Time_17_09_2023__23_34_19/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnet50"
        encoder_name2 = "efficientnet-b7"
    elif selected == "efficientnet b5 s2":
        # efficientnet b5 stage 2 used twice
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_Time_15_09_2023__21_19_19/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_Time_15_09_2023__21_19_19/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "resnet50":
        # resnet50
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnet50_Time_30_08_2023__10_37_43/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_resnet50_Time_16_09_2023__00_58_50/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnet50"
    elif selected == "resnet101":
        # resnet101
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnet101_Time_30_08_2023__14_51_07/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_resnet101_Time_16_09_2023__04_07_08/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnet101"
    elif selected == "resnet152":
        # resnet152
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnet152_Time_30_08_2023__19_17_27/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_resnet152_Time_16_09_2023__07_18_34/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnet152"
    elif selected == "resnext50_32x4d":
        # resnet152
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnext50_32x4d_Time_31_08_2023__00_17_35/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_resnext50_32x4d_Time_16_09_2023__10_43_18/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnext50_32x4d"
    elif selected == "resnext101_32x16d":
        # resnet152
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnext101_32x16d_Time_09_09_2023__00_54_46/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_resnext101_32x16d_Time_16_09_2023__14_36_43/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnext101_32x16d"
    elif selected == "resnext101_32x48d":
        # resnet152
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_resnext101_32x48d_Time_01_09_2023__09_16_19/models/epoch400_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_resnext101_32x48d_Time_16_09_2023__21_33_52/models/epoch400_Testdice0.0000.pkl'
        encoder_name = "resnext101_32x48d"
    elif selected == "20S":
        # 20 samples train only
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_20S_Time_19_09_2023__08_07_37/models/epoch100_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_20S_Time_19_09_2023__08_32_38/models/epoch100_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "20F":
        # 20 samples train only
        #weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_20F_Time_18_09_2023__23_35_56/models/epoch100_Testdice0.0000.pkl'
        #weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_20F_Time_19_09_2023__00_00_49/models/epoch100_Testdice0.0000.pkl'
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_20F_Time_19_09_2023__18_51_06/models/epoch100_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_20F_Time_19_09_2023__19_20_07/models/epoch100_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "40S":
        # 40 samples train only
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_40S_Time_19_09_2023__09_29_37/models/epoch120_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_40S_Time_19_09_2023__09_29_37/models/epoch120_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "40F":
        # 20 samples train only
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_40F_Time_19_09_2023__00_27_25/models/epoch120_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_40F_Time_19_09_2023__00_57_49/models/epoch120_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "100S":
        # 100 samples train only
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_100S_Time_19_09_2023__10_00_59/models/epoch200_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_100S_Time_19_09_2023__10_51_49/models/epoch200_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "100F":
        # 100 samples train only
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_100F_Time_19_09_2023__01_30_29/models/epoch200_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_100F_Time_19_09_2023__02_24_28/models/epoch200_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "200S":
        # 200 samples train only
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_200S_Time_19_09_2023__11_52_57/models/epoch200_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_200S_Time_19_09_2023__12_55_08/models/epoch200_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"
    elif selected == "200F":
        # 200 samples train only
        weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_efficientnet-b5_200F_Time_19_09_2023__03_26_22/models/epoch200_Testdice0.0000.pkl'
        weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_efficientnet-b5_200F_Time_19_09_2023__04_28_07/models/epoch200_Testdice0.0000.pkl'
        encoder_name = "efficientnet-b5"

    c1_size = 256
    c1_tta = True
    orimg = False

    use_c2_flag = True
    c2_tta = True
    c2_size = 256
    c2_resize_order = 0
    # GPU
    cuda_idx = 0


# ===============================================================================================

    # Set GPU related
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)



    # ===============
    device = torch.device('cuda')
    # read test set
    if fold_flag:
        _, test = get_fold_filelist(csv_file, K=fold_K, fold=fold_index)
        test_img_list = [img_path+sep+i[0] for i in test]
        if mask_path is not None:
            test_mask_list = [mask_path+sep+i[0] for i in test]
    else:
        test_img_list = get_filelist_frompath(img_path,'PNG')
        if mask_path is not None:
            test_mask_list = [mask_path + sep + i.split(sep)[-1] for i in test_img_list]

    # build two models
    with torch.no_grad():
        # tta set up
        tta_trans = tta.Compose([
            tta.VerticalFlip(),
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0,180]),
        ])
        # build model
        # cascade1
        model_cascade1 = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, in_channels=1, classes=1)
        model_cascade1.to(device)
        model_cascade1.load_state_dict(torch.load(weight_c1))
        if c1_tta:
            model_cascade1 = tta.SegmentationTTAWrapper(model_cascade1, tta_trans,merge_mode='mean')
        model_cascade1.eval()
        # cascade2
        if encoder_name2 == "Null":
            encoder_name2 = encoder_name
        model_cascade2 = smp.DeepLabV3Plus(encoder_name=encoder_name2, encoder_weights=None, in_channels=1, classes=1)
        model_cascade2.to(device)
        model_cascade2.load_state_dict(torch.load(weight_c2))
        if c2_tta:
            model_cascade2 = tta.SegmentationTTAWrapper(model_cascade2, tta_trans,merge_mode='mean')
        model_cascade2.eval()


        # index
        IOU_list = []
        DSC_list = []
        ioumin3 = 0

        #
        for index, img_file in enumerate(test_img_list):
            print(task_name,'\n',img_file,index+1,'/',len(test_img_list),' c1:',c1_size,' c2: ', c2_size)
            with torch.no_grad():
                # Read GT data
                if mask_path is not None:
                    mask_file = test_mask_list[index]
                    GT = Image.open(mask_file)
                    Transform_GT = T.Compose([T.ToTensor()])
                    GT = Transform_GT(GT)
                    GT_array = (torch.squeeze(GT)).data.cpu().numpy()



                # Process the original image
                img, cut_image_orshape, or_shape, location = TNSCUI_preprocess(img_file,outputsize=c1_size,orimg=orimg)
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)
                img = img.to(device)
                img_array = (torch.squeeze(img)).data.cpu().numpy()

                """go through cascade1"""
                # get cascade1 output
                with torch.no_grad():
                    mask_c1 = model_cascade1(img)
                    mask_c1 = torch.sigmoid(mask_c1)
                    mask_c1_array = (torch.squeeze(mask_c1)).data.cpu().numpy()
                    mask_c1_array = (mask_c1_array>0.5)
                    mask_c1_array = mask_c1_array.astype(np.float32)
                    # Get the largest Unicom domain
                    mask_c1_array_biggest = largestConnectComponent(mask_c1_array.astype(np.int))
                    print("c1 outputsize:",mask_c1_array_biggest.shape)



                """go through cascade2"""
                with torch.no_grad():
                    if use_c2_flag:
                        # Get the bounding box coordinates of roi
                        dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max = preprocess(mask_c1_array_biggest,c1_size)
                        # Get the img area according to the bounding box coordinates of roi
                        img_array_roi = img_array[dim1_cut_min:dim1_cut_max,dim2_cut_min:dim2_cut_max]
                        img_array_roi_shape = img_array_roi.shape
                        img_array_roi = resize(img_array_roi, (c2_size, c2_size), order=3)
                        img_array_roi_tensor = torch.tensor(data = img_array_roi,dtype=img.dtype)
                        img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor,0)
                        img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor,0).to(device)
                        # Get cascade2 output, and restore the size
                        print('use cascade2')
                        mask_c2 = model_cascade2(img_array_roi_tensor)
                        mask_c2 = torch.sigmoid(mask_c2)
                        mask_c2_array = (torch.squeeze(mask_c2)).data.cpu().numpy()
                        if saveas == 'mask':
                            cascade2_t = 0.5
                            mask_c2_array = (mask_c2_array>cascade2_t)
                            print(cascade2_t)
                        mask_c2_array = mask_c2_array.astype(np.float32)
                        mask_c2_array = resize(mask_c2_array, img_array_roi_shape, order=c2_resize_order)
                        # Put back the mask output by cascade1
                        mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                        mask_c1_array_biggest[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max] = mask_c2_array
                        mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                        print(np.unique(mask_c1_array_biggest))
                        print(np.sum(mask_c1_array_biggest))


                # According to the preprocessing information, first restore to the original size, and then put it back to the original image position
                mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                final_mask = np.zeros(shape=or_shape, dtype=mask_c1_array_biggest.dtype)
                mask_c1_array_biggest = resize(mask_c1_array_biggest, cut_image_orshape, order=1)
                final_mask[location[0]:location[1],location[2]:location[3]]=mask_c1_array_biggest



                # into a binary image
                if saveas == 'mask':
                    final_mask = (final_mask > 0.5)
                    # final_mask = (final_mask > 0.05)
                final_mask = final_mask.astype(np.float32)
                print(np.unique(final_mask))
                print(np.sum(final_mask))


                # If there is GT, calculate the index
                if mask_path is not None:
                    if getIOU(final_mask, GT_array)<0.3:
                        ioumin3 = ioumin3+1
                    IOU_list.append(getIOU(final_mask, GT_array))
                    print('IOU:',getIOU(final_mask, GT_array))
                    IOU_final = np.mean(IOU_list)
                    print('fold:',fold_index,'  IOU_final',IOU_final)

                    DSC_list.append(getDSC(final_mask, GT_array))
                    print('DSC:',getDSC(final_mask, GT_array))
                    DSC_final = np.mean(DSC_list)
                    print('fold:',fold_index,'  DSC_final',DSC_final)

                # save image
                if save_path is not None:
                    final_mask = final_mask*255
                    final_mask = final_mask.astype(np.uint8)
                    print(np.unique(final_mask),'\n')
                    # print(np.max())
                    final_savepath = save_path+sep+img_file.split(sep)[-1]
                    im = Image.fromarray(final_mask)
                    im.save(final_savepath)



            #
            # plt.subplot(1, 2, 1)
            # plt.imshow(mask_c1_array_biggest,cmap=plt.cm.gray)
            # # plt.imshow(final_mask,cmap=plt.cm.gray)
            # plt.subplot(1, 2, 2)
            # plt.imshow(img_array,cmap=plt.cm.gray)
            # plt.show()


    print((ioumin3))
    input()
