import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import fftpack
from PIL import Image


# Low Pass Filter
def LPF(img, keep):
    r, c = img.shape
    img_lpf = img.copy()
    img_lpf[int(r * keep):int(r * (1 - keep))] = 0
    img_lpf[:, int(c * keep):int(c * (1 - keep))] = 0
    return img_lpf


# High Pass Filter
def HPF(img, keep):
    r, c = img.shape
    img_hpf = img.copy()
    img_hpf[0:int(r * keep), 0:int(c * keep)] = 0
    img_hpf[int(r * (1 - keep)):r, int(c * (1 - keep)):c] = 0
    img_hpf[0:int(r * keep), int(c * (1 - keep)):c] = 0
    img_hpf[int(r * (1 - keep)):r, 0:int(c * keep)] = 0
    return img_hpf


# Non Linear Exponential high frequency enhancement filter
def HPMag(img, power):
    r, c = img.shape
    imgMask = np.zeros_like(img)
    img_hpm = img.copy()
    #Below image creation is only when images are of different sizes, otherwise we can prepare the enhanceent matrix before
    for row in range(r):
        for col in range(c):
            imgMask[row,col] = ((r/2-abs(row-(r/2)))**power) * ((c/2-abs(col-(c/2)))**power)
    img_hpm = img_hpm * imgMask
    return img_hpm


# DC removal (frequency=0)
def DC_remove(img):
    r, c = img.shape
    img_dc = img.copy()
    img_dc[0, 0] = 0
    img_dc[r - 1, 0] = 0
    img_dc[0, c - 1] = 0
    img_dc[r - 1, c - 1] = 0
    return img_dc


def Array2Img(imgArr):
    image = imgArr - imgArr.min()
    image *= (255.0 / image.max())
    image = image.astype(np.uint8)
    img = Image.fromarray(image)
    return img


# Below functions perform frequency filters on batches of images
def lpf_images(images, keep): #, random_state, parents, hooks):
    im = np.array(images).astype(float)
    if len(im.shape)==2:
        im_fft = fftpack.fft2(im)
        im_fft2 = LPF(im_fft, keep)
        im_ifft = fftpack.ifft2(im_fft2).real
        #image_aug = Array2Img(im_ifft)
        result = im_ifft
    else:
        result = []
        for image in images:
            im = np.array(image).astype(float)
            im_fft = fftpack.fft2(im)
            im_fft2 = LPF(im_fft, keep)
            im_ifft = fftpack.ifft2(im_fft2).real
            #image_aug = Array2Img(im_ifft)
            result.append(im_ifft)
    return result


def hpf_images(images, keep): #random_state, parents, hooks):
    im = np.array(images).astype(float)
    if len(im.shape)==2:
        im_fft = fftpack.fft2(im)
        im_fft2 = HPF(im_fft, keep)
        im_ifft = fftpack.ifft2(im_fft2).real
        #image_aug = Array2Img(im_ifft)
        result = im_ifft
    else:
        result = []
        for image in images:
            im = np.array(image).astype(float)
            im_fft = fftpack.fft2(im)
            im_fft2 = HPF(im_fft, keep)
            im_ifft = fftpack.ifft2(im_fft2).real
            #image_aug = Array2Img(im_ifft)
            result.append(im_ifft)
    return result


def hpmag_images(images, power): #random_state, parents, hooks):
    im = np.array(images).astype(float)
    if len(im.shape)==2:
        im_fft = fftpack.fft2(im)
        im_fft2 = HPMag(im_fft, power)
        im_ifft = fftpack.ifft2(im_fft2).real
        #image_aug = Array2Img(im_ifft)
        result = im_ifft
    else:
        result = []
        for image in images:
            im = np.array(image).astype(float)
            im_fft = fftpack.fft2(im)
            im_fft2 = HPMag(im_fft, power)
            im_ifft = fftpack.ifft2(im_fft2).real
            #image_aug = Array2Img(im_ifft)
            result.append(im_ifft)
    return result


def dc_images(images, keep): #, random_state, parents, hooks):
    im = np.array(images).astype(float)
    if len(im.shape)==2:
        r, c = im.shape
        im_fft = fftpack.fft2(im)
        im_fft[0, 0] =         im_fft[0, 0] * keep
        im_fft[r - 1, 0] =     im_fft[r - 1, 0] * keep
        im_fft[0, c - 1] =     im_fft[0, c - 1] * keep
        im_fft[r - 1, c - 1] = im_fft[r - 1, c - 1] * keep
        im_ifft = fftpack.ifft2(im_fft).real
        #image_aug = Array2Img(im_ifft)
        result = im_ifft
    else:
        result = []
        for image in images:
            r, c = image.shape
            im = np.array(image).astype(float)
            im_fft = fftpack.fft2(im)
            im_fft[0, 0] = im_fft[0, 0] * keep
            im_fft[r - 1, 0] = im_fft[r - 1, 0] * keep
            im_fft[0, c - 1] = im_fft[0, c - 1] * keep
            im_fft[r - 1, c - 1] = im_fft[r - 1, c - 1] * keep
            im_ifft = fftpack.ifft2(im_fft).real
            #image_aug = Array2Img(im_ifft)
            result.append(im_ifft)
    return result


def data_aug(imgs, masks, segs=None):  # Input images and labels, the input and output format is numpy
    # standardized format
    imgs = np.array(imgs)
    masks = np.array(masks).astype(np.uint8)
    if segs is not None:
        segs = np.array(segs).astype(np.uint8)

    # print('imgs shape',imgs.shape)
    # Determine the number of batches
    # if imgs.shape == (256, 256):
    if len(imgs.shape) == 2:  # Determine whether the image is two-dimensional, so as to determine whether it is a batch
        batch_num = 1
    else:
        batch_num = 1

    # We define function called sometimes, doing augmentation in specific probability
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # Set random function, 50% chance to increase,or
    # sometimes = lambda aug: iaa.Sometimes(0.7, aug)     # Set random function, 70% chance to increase

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # 50% of the image is flipped horizontally
            iaa.Flipud(0.5),  # 50% of the image is flipped vertically

            sometimes(iaa.Crop(percent=(0, 0.1))),
            # Perform a crop operation on a random part of the image. The crop range is 0 to 10%
            # sometimes(iaa.Crop(percent=(0, 0.2))),  # Perform a crop operation on a random part of the image. The crop range is 0 to 20% wang

            sometimes(iaa.Affine(  # Affine transformation of part of the image
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Image scaling between 80% and 120%
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translation between ±20%
                rotate=(-45, 45),  # Rotate between ±45 degrees
                shear=(-16, 16),  # Shear transformation ±16 degrees, (rectangle becomes parallelogram)
                order=[0, 1],  # Use nearest neighbor difference or bilinear difference
                cval=(0, 255),
                mode=ia.ALL,  # edge padding
            )),

            # Use the following methods between 0 and 5 to enhance the image
            iaa.SomeOf((0, 5),
                       [
                           # sharpening
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)),

                           # Warp local regions of the image
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                           # change contrast
                           iaa.contrast.LinearContrast((0.75, 1.25), per_channel=0.5),

                           # Use one of Gaussian blur, mean blur, and median blur
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Gaussian noise added
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Edge detection, assign the detected value to 0 or 255 and then superimpose it on the original image (not sure)
                           # sometimes(iaa.OneOf([
                           #     iaa.EdgeDetect(alpha=(0, 0.7)),
                           #     iaa.DirectedEdgeDetect(
                           #         alpha=(0, 0.7), direction=(0.0, 1.0)
                           #     ),
                           # ])),

                           # Relief effect (very strange operation, not sure if it can be used)
                           # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Set 1% to 10% of the pixels to black or cover 3% to 15% of the pixels with a black square 2% to 5% of the original image size
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               )
                           ]),

                           # Move pixels around. This method is seen in the mnist dataset enhancement
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                       ],

                       random_order=True  # Apply these operations to the image in random order
                       )
        ],
        random_order=True  # Apply these operations to the image in random order

    )

    # freqAug is a selecting wether we will do frequency augmentation to data as well
    freqAug = 1
    if batch_num > 1:  # Multi-batch processing
        images_aug = np.zeros(imgs.shape)
        segmaps_aug = np.zeros(masks.shape)
        if segs is not None:
            segmaps_aug_seg = np.zeros(masks.shape)
        freq_prob = 2
        random_filt1 = np.random.random(1) * freq_prob / 3 * freqAug
        random_filt2 = np.random.random(1) * freq_prob * freqAug
        random_filt3 = np.random.random(1) * freq_prob / 3 * freqAug
        random_filt4 = np.random.random(1) * freq_prob * 10 * freqAug
        for i in range(0, batch_num):
            seq_det = seq.to_deterministic()  # Determine a data augmentation sequence
            images_aug[i, :, :] = seq_det.augment_images(imgs[i, :, :])  # make enhancements


            if random_filt1<0.2 and random_filt1>0.05:
                images_aug[i, :, :] = hpmag_images(images_aug[i, :, :], power=random_filt1)
            #elif random_filt3<0.01 and random_filt3>0.001:
            #    images_aug[i, :, :] = hpf_images(images_aug[i, :, :], keep=0.01)
            if random_filt2<0.5 and random_filt2>0.07:
                images_aug[i, :, :] = lpf_images(images_aug[i, :, :], keep=random_filt2)
            if random_filt4 < 3 and random_filt4 > 0.2:
                images_aug[i, :, :] = dc_images(images_aug[i, :, :], keep=random_filt4)

            segmap = ia.SegmentationMapsOnImage(masks[i, :, :], shape=masks[i, :, :].shape)  # split tag format
            segmaps_aug[i, :, :] = seq_det.augment_segmentation_maps(segmap).get_arr().astype(
                np.uint8)  # Apply the method to the segmentation label and convert it to np type
            if segs is not None:
                segmap_seg = ia.SegmentationMapsOnImage(segs[i, :, :], shape=segs[i, :, :].shape)  # split tag format
                segmaps_aug_seg[i, :, :] = seq_det.augment_segmentation_maps(segmap_seg).get_arr().astype(np.uint8)

    else:
        seq_det = seq.to_deterministic()  # Determine a data augmentation sequence
        # print('imgs.shape',imgs.shape)
        images_aug = seq_det.augment_images(imgs)  # make enhancements

        freq_prob = 2
        random_filt1 = np.random.random(1) * freq_prob / 3 * freqAug
        random_filt2 = np.random.random(1) * freq_prob * freqAug
        random_filt3 = np.random.random(1) * freq_prob / 3 * freqAug
        random_filt4 = np.random.random(1) * freq_prob * 10 * freqAug
        if random_filt1 < 0.2 and random_filt1 > 0.05:
            images_aug = hpmag_images(images_aug, power=random_filt1)
        #elif random_filt3 < 0.01 and random_filt3 > 0.001:
        #    images_aug = hpf_images(images_aug, keep=0.01)
        if random_filt2 < 0.5 and random_filt2 > 0.07:
            images_aug = lpf_images(images_aug, keep = random_filt2)
        if random_filt4 < 3 and random_filt4 > 0.2:
            images_aug = dc_images(images_aug, keep = random_filt4)

        segmaps = ia.SegmentationMapsOnImage(masks, shape=masks.shape)  # split tag format
        segmaps_aug = seq_det.augment_segmentation_maps(segmaps).get_arr().astype(
            np.uint8)  # Apply the method to the segmentation label and convert it to np type
        if segs is not None:
            segmap_seg = ia.SegmentationMapsOnImage(segs, shape=segs.shape)  # split tag format
            segmaps_aug_seg = seq_det.augment_segmentation_maps(segmap_seg).get_arr().astype(np.uint8)

    if segs is not None:
        return images_aug, segmaps_aug, segmaps_aug_seg
    else:
        return images_aug, segmaps_aug
