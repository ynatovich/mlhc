# Calculate dice one by one, and then save
# Calculating dice is not the ultimate goal, this script should be used as the final image output script
import torch
import ttach as tta
from PIL import Image
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.transform import resize
from torchvision import transforms as T
import torch.nn.functional as F

import segmentation_models_pytorch_4TorchLessThan120 as smp
from tnscui_utils.TNSCUI_preprocess import TNSCUI_preprocess
from tnscui_utils.TNSUCI_util import *

from AutoSAM.segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from AutoSAM.models import sam_seg_model_registry


def getIOU(SR, GT):
    """
    are binary images
    :param SR: 
    :param GT: 
    :return: 
    """
    TP = (SR + GT == 2).astype(np.float32)
    FP = (SR + (1 - GT) == 2).astype(np.float32)
    FN = ((1 - SR) + GT == 2).astype(np.float32)

    IOU = float(np.sum(TP)) / (float(np.sum(TP + FP + FN)) + 1e-6)

    return IOU


def getDSC(SR, GT):
    """
    are binary images
    :param SR: 
    :param GT: 
    :return: 
    """

    Inter = np.sum(((SR + GT) == 2).astype(np.float32))
    DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

    return DC


def largestConnectComponent(bw_img):
    if np.sum(bw_img) == 0:
        return bw_img
    labeled_img, num = sklabel(bw_img, background=0, connectivity=1, return_num=True)
    if num == 1:
        return bw_img

    max_label = 0
    max_num = 0
    for i in range(0, num):
        print(i)
        if np.sum(labeled_img == (i + 1)) > max_num:
            max_num = np.sum(labeled_img == (i + 1))
            max_label = i + 1
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

    max_lengthl = int((c1_size / 256) * 80)
    preprocess1 = int((c1_size / 256) * 19)
    pp22 = int((c1_size / 256) * 31)

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
    return [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]


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
    saveas = 'mask'  # Save in the form of prob probability graph, save in the form of mask binary graph
    # 路径设置
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

    weight_c1 = r'./result/TNSCUI/dpv3plus_stage1_5_1_Time_04_09_2023__22_45_46/models/epoch400_Testdice0.0000.pkl'
    weight_c2 = r'./result/TNSCUI/dpv3plus_stage2_5_1_Time_05_09_2023__00_53_28/models/epoch400_Testdice0.0000.pkl'

    autosam_path = r'./AutoSAM/scripts/model_best_sample_num_10.pth.tar'
    sam_path = r'./sam/sam_vit_h_4b8939.pth'

    c1_size = 256
    c1_tta = False
    orimg = False


    # ===============
    device = torch.device('cuda')
    # read test set
    if fold_flag:
        _, test = get_fold_filelist(csv_file, K=fold_K, fold=fold_index)
        test_img_list = [img_path + sep + i[0] for i in test]
        if mask_path is not None:
            test_mask_list = [mask_path + sep + i[0] for i in test]
    else:
        test_img_list = get_filelist_frompath(img_path, 'PNG')
        if mask_path is not None:
            test_mask_list = [mask_path + sep + i.split(sep)[-1] for i in test_img_list]

    # build two models
    with torch.no_grad():
        # tta设置
        tta_trans = tta.Compose([
            tta.VerticalFlip(),
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
        ])


        autosam_model = sam_seg_model_registry['vit_h'](num_classes=2, checkpoint=autosam_path)
        autosam_model.cuda()
        autosam_model.eval()

        # index
        IOU_list = []
        DSC_list = []
        ioumin3 = 0

        #
        for index, img_file in enumerate(test_img_list):
            print(task_name, '\n', img_file, index + 1, '/', len(test_img_list), ' c1:', c1_size)
            with torch.no_grad():
                # Read GT data
                if mask_path is not None:
                    mask_file = test_mask_list[index]
                    GT = Image.open(mask_file)
                    Transform_GT = T.Compose([T.ToTensor()])
                    GT = Transform_GT(GT)
                    GT_array = (torch.squeeze(GT)).data.cpu().numpy()

                # Process the original image
                img, cut_image_orshape, or_shape, location = TNSCUI_preprocess(img_file, outputsize=c1_size,
                                                                               orimg=orimg)
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)
                img = img.to(device)
                b, c, h, w = img.shape
                img_uint8 = img.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
                img_uint8 = (img_uint8 * 255).astype(np.uint8)

                img_array = (torch.squeeze(img)).data.cpu().numpy()

                """过一遍cascade1"""
                # get cascade1 output
                with torch.no_grad():
                    autosam_mask, iou_pred = autosam_model(img.repeat(1, 3, 1, 1))
                    top_autosam_mask = autosam_mask[torch.argmax(iou_pred)]
                    autosam_mask = autosam_mask.view(b, -1, h, w)
                    softmax_output = F.softmax(autosam_mask, dim=1)
                    autosam_mask = torch.argmax(softmax_output, dim=1)

                autosam_mask = autosam_mask.cpu().numpy().astype(np.float32)
                final_mask = np.zeros(shape=or_shape, dtype=autosam_mask.dtype)
                autosam_mask = resize(autosam_mask.squeeze(), cut_image_orshape, order=1)
                final_mask[location[0]:location[1], location[2]:location[3]] = autosam_mask

                # into a binary image
                if saveas == 'mask':
                    final_mask = (final_mask > 0.5)
                    # final_mask = (final_mask > 0.05)
                final_mask = final_mask.astype(np.float32)
                print(np.unique(final_mask))
                print(np.sum(final_mask))

                # If there is GT, calculate the index
                if mask_path is not None:
                    if getIOU(final_mask, GT_array) < 0.3:
                        ioumin3 = ioumin3 + 1
                    IOU_list.append(getIOU(final_mask, GT_array))
                    print('IOU:', getIOU(final_mask, GT_array))
                    IOU_final = np.mean(IOU_list)
                    print('fold:', fold_index, '  IOU_final', IOU_final)

                    DSC_list.append(getDSC(final_mask, GT_array))
                    print('DSC:', getDSC(final_mask, GT_array))
                    DSC_final = np.mean(DSC_list)
                    print('fold:', fold_index, '  DSC_final', DSC_final)

                # save image
                if save_path is not None:
                    final_mask = final_mask * 255
                    final_mask = final_mask.astype(np.uint8)
                    print(np.unique(final_mask), '\n')
                    # print(np.max())
                    final_savepath = save_path + sep + img_file.split(sep)[-1]
                    im = Image.fromarray(final_mask)
                    im.save(final_savepath)


    print((ioumin3))
    input()
