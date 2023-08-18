import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm
import os
from os import listdir
from PIL import Image

# get the path/directory
image1_load_dir = './2_preprocessed_data/stage1/p_image/'
mask1_load_dir = './2_preprocessed_data/stage1/p_mask/'
image1_save_dir = './3_preprocessed_data/stage1/p_image/'
mask1_save_dir = './3_preprocessed_data/stage1/p_mask/'
image2_load_dir = './2_preprocessed_data/stage2/p_image/'
mask2_load_dir = './2_preprocessed_data/stage2/p_mask/'
image2_save_dir = './3_preprocessed_data/stage2/p_image/'
mask2_save_dir = './3_preprocessed_data/stage2/p_mask/'

# for LPF
keep_fraction1 = 0.2
# for HPF
keep_fraction2 = 0.01

def LPF(img, keep):
    r, c = img.shape
    img_lpf = img.copy()
    img_lpf[int(r * keep):int(r * (1 - keep))] = 0
    img_lpf[:, int(c * keep):int(c * (1 - keep))] = 0
    return img_lpf

def HPF(img, keep):
    r, c = img.shape
    img_hpf = img.copy()
    img_hpf[0:int(r * keep_fraction2), 0:int(c * keep_fraction2)] = 0
    img_hpf[int(r * (1 - keep_fraction2)):r, int(c * (1 - keep_fraction2)):c] = 0
    img_hpf[0:int(r * keep_fraction2), int(c * (1 - keep_fraction2)):c] = 0
    img_hpf[int(r * (1 - keep_fraction2)):r, 0:int(c * keep_fraction2)] = 0
    return img_hpf

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

def plotImg(imgIn, imgFilt):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(imgIn, plt.cm.gray)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(imgFilt, plt.cm.gray)
    plt.title("Filtered")
    plt.show()


for images in os.listdir(image1_load_dir):
    # check if the image ends with png
    if (images.endswith(".PNG")):
        load_image = image1_load_dir + images
        img_orig = Image.open(load_image)
        im = np.array(img_orig).astype(float)

        im_fft = fftpack.fft2(im)
        im_fft2a = LPF(im_fft, keep_fraction1)
        im_fft2b = HPF(im_fft, keep_fraction2)
        im_fft2 = im_fft2b + im_fft2a
        im_fft2 = DC_remove(im_fft2)
        im_ifft = fftpack.ifft2(im_fft2).real

        imgFilt = Array2Img(im_ifft)

        plotImg(img_orig, imgFilt)

        save_image = image1_save_dir + 'fft' + images
        imgFilt.save(save_image)  # Saves the modified pixels to image

        load_mask = mask1_load_dir + images
        mask_orig = Image.open(load_mask)
        save_mask = mask1_save_dir + 'fft' + images
        mask_orig.save(save_mask)  # Saves the modified pixels to image



for images in os.listdir(image2_load_dir):
    # check if the image ends with png
    if (images.endswith(".PNG")):
        load_image = image2_load_dir + images
        img_orig = Image.open(load_image)
        im = np.array(img_orig).astype(float)

        im_fft = fftpack.fft2(im)
        im_fft2a = LPF(im_fft, keep_fraction1)
        im_fft2b = HPF(im_fft, keep_fraction2)
        im_fft2 = im_fft2b + im_fft2a
        im_fft2 = DC_remove(im_fft2)
        im_ifft = fftpack.ifft2(im_fft2).real

        imgFilt = Array2Img(im_ifft)

        plotImg(img_orig, imgFilt)

        save_image = image2_save_dir + 'fft' + images
        imgFilt.save(save_image)  # Saves the modified pixels to image

        load_mask = mask2_load_dir + images
        mask_orig = Image.open(load_mask)
        save_mask = mask2_save_dir + 'fft' + images
        mask_orig.save(save_mask)  # Saves the modified pixels to image

