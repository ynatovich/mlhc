import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm
from PIL import Image
from matplotlib import cm

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

def HPMag(img, power):
    r, c = img.shape
    imgMask = np.zeros_like(img)
    img_hpm = img.copy()
    for row in range(r):
        for col in range(c):
            #imgMask[row,col] = ((r/2-abs(row-(r/2)))**power) * ((c/2-abs(col-(c/2)))**power)
            imgMask[row,col] = (((r/2-abs(row-(r/2)))**2) + ((c/2-abs(col-(c/2)))**2)) ** power
    img_hpm = img_hpm * imgMask
    return img_hpm

def DC_remove(img,keep):
    r, c = img.shape
    img_dc = img.copy()
    img_dc[0, 0] = img_dc[0, 0] * keep
    img_dc[r - 1, 0] = img_dc[r - 1, 0] * keep
    img_dc[0, c - 1] = img_dc[0, c - 1] * keep
    img_dc[r - 1, c - 1] = img_dc[r - 1, c - 1] * keep
    return img_dc

def filter(img,type,param):
    img_out = img.copy()
    if type == 'LPF':
        img_out = LPF(img, param)
    if type == 'HPF':
        img_out = HPF(img, param)
    if type == 'HPM':
        img_out = HPMag(img, param)
    if type == 'None':
        img_out = np.zeros_like(img)
    return img_out


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





im_p1     = plt.imread('./2_preprocessed_data/stage1/p_image/1.PNG').astype(float)
im_p10    = plt.imread('./2_preprocessed_data/stage1/p_image/10.PNG').astype(float)
im_p100   = plt.imread('./2_preprocessed_data/stage1/p_image/100.PNG').astype(float)
im_p10004 = plt.imread('./2_preprocessed_data/stage1/p_image/10004.PNG').astype(float)

im_m1     = plt.imread('./2_preprocessed_data/stage1/p_mask/1.PNG').astype(float)
im_m10    = plt.imread('./2_preprocessed_data/stage1/p_mask/10.PNG').astype(float)
im_m100   = plt.imread('./2_preprocessed_data/stage1/p_mask/100.PNG').astype(float)
im_m10004 = plt.imread('./2_preprocessed_data/stage1/p_mask/10004.PNG').astype(float)

im_p1_fft     = fftpack.fft2(im_p1)
im_p10_fft    = fftpack.fft2(im_p10)
im_p100_fft   = fftpack.fft2(im_p100)
im_p10004_fft = fftpack.fft2(im_p10004)

if False:
    r, c = im_p1.shape
    imgMask = np.zeros_like(im_p1)
    img_hpm = im_p1.copy()
    power = 2
    for row in range(r):
        for col in range(c):
            # imgMask[row,col] = ((r/2-abs(row-(r/2)))**power) * ((c/2-abs(col-(c/2)))**power)
            imgMask[row, col] = (((r / 2 - abs(row - (r / 2))) ** 2) + ((c / 2 - abs(col - (c / 2))) ** 2)) ** power

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(0, 256, 1)
    Y = np.arange(0, 256, 1)
    X, Y = np.meshgrid(X, Y)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, np.abs(imgMask), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(0, 20)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# In the lines following, we'll make a copy of the original spectrum and
# truncate coefficients.
# Define the fraction of coefficients (in each direction) we keep
# for LPF
keep_fraction1 = 0.07
# for HPF
keep_fraction2 = 0.01
# Power
power = 0.5
# DC enhance
dcParam = 0

RGB = False

# Call ff a copy of the original transform. Numpy arrays have a copy
# method for this purpose.
im_p1_fft2a     = im_p1_fft.copy()
im_p10_fft2a    = im_p10_fft.copy()
im_p100_fft2a   = im_p100_fft.copy()
im_p10004_fft2a = im_p10004_fft.copy()
# Set r and c to be the number of rows and columns of the array.


filter_type = 'HPM'
param = power #keep_fraction1, keep_fraction1

im_p1_fft2a = filter(im_p1_fft2a, filter_type, param)
im_p10_fft2a = filter(im_p10_fft2a, filter_type, param)
im_p100_fft2a = filter(im_p100_fft2a, filter_type, param)
im_p10004_fft2a = filter(im_p10004_fft2a, filter_type, param)

filter_type = 'LPF'
param = keep_fraction1

im_p1_fft2b = filter(im_p1_fft2a, filter_type, param)
im_p10_fft2b = filter(im_p10_fft2a, filter_type, param)
im_p100_fft2b = filter(im_p100_fft2a, filter_type, param)
im_p10004_fft2b = filter(im_p10004_fft2a, filter_type, param)


im_p1_fft2     = DC_remove(im_p1_fft2b, dcParam)
im_p10_fft2    = DC_remove(im_p10_fft2b, dcParam)
im_p100_fft2   = DC_remove(im_p100_fft2b, dcParam)
im_p10004_fft2 = DC_remove(im_p10004_fft2b, dcParam)


# Now reconstruct
im_p1_ifft     = fftpack.ifft2(im_p1_fft2).real
im_p10_ifft    = fftpack.ifft2(im_p10_fft2).real
im_p100_ifft   = fftpack.ifft2(im_p100_fft2).real
im_p10004_ifft = fftpack.ifft2(im_p10004_fft2).real

if RGB==True:
    im_p1_iffta     = fftpack.ifft2(im_p1_fft2a).real
    im_p10_iffta    = fftpack.ifft2(im_p10_fft2a).real
    im_p100_iffta   = fftpack.ifft2(im_p100_fft2a).real
    im_p10004_iffta = fftpack.ifft2(im_p10004_fft2a).real
    im_p1_ifft      = np.dstack((im_p1,im_p1_ifft,im_p1_iffta))
    im_p10_ifft     = np.dstack((im_p10,im_p10_ifft,im_p10_iffta))
    im_p100_ifft    = np.dstack((im_p100,im_p100_ifft,im_p100_iffta))
    im_p10004_ifft  = np.dstack((im_p10004,im_p10004_ifft,im_p10004_iffta))

im_p1_ifft     = Array2Img(im_p1_ifft)
im_p10_ifft    = Array2Img(im_p10_ifft)
im_p100_ifft   = Array2Img(im_p100_ifft)
im_p10004_ifft = Array2Img(im_p10004_ifft)

plt.figure()
plt.subplot(5, 4, 1)
plt.imshow(im_p1, plt.cm.gray)
plt.title('image 1')

plt.subplot(5, 4, 2)
plt.imshow(im_p10, plt.cm.gray)
plt.title('image 10')

plt.subplot(5, 4, 3)
plt.imshow(im_p100, plt.cm.gray)
plt.title('image 100')

plt.subplot(5, 4, 4)
plt.imshow(im_p10004, plt.cm.gray)
plt.title('image 10004')

plt.subplot(5, 4, 5)
plt.imshow(im_m1, plt.cm.gray)
plt.title('mask 1')

plt.subplot(5, 4, 6)
plt.imshow(im_m10, plt.cm.gray)
plt.title('mask 10')

plt.subplot(5, 4, 7)
plt.imshow(im_m100, plt.cm.gray)
plt.title('mask 100')

plt.subplot(5, 4, 8)
plt.imshow(im_m10004, plt.cm.gray)
plt.title('mask 10004')

plt.subplot(5, 4, 9)
plt.imshow(np.abs(im_p1_fft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft 1')

plt.subplot(5, 4, 10)
plt.imshow(np.abs(im_p10_fft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft 10')

plt.subplot(5, 4, 11)
plt.imshow(np.abs(im_p100_fft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft 100')

plt.subplot(5, 4, 12)
plt.imshow(np.abs(im_p10004_fft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft 10004')

plt.subplot(5, 4, 13)
plt.imshow(np.abs(im_p1_fft2), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft filtered 1')

plt.subplot(5, 4, 14)
plt.imshow(np.abs(im_p10_fft2), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft filtered 10')

plt.subplot(5, 4, 15)
plt.imshow(np.abs(im_p100_fft2), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft filtered 100')

plt.subplot(5, 4, 16)
plt.imshow(np.abs(im_p10004_fft2), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('fft filtered 10004')

plt.subplot(5, 4, 17)
plt.imshow(im_p1_ifft, plt.cm.gray)
plt.title('image 1')

plt.subplot(5, 4, 18)
plt.imshow(im_p10_ifft, plt.cm.gray)
plt.title('image 10')

plt.subplot(5, 4, 19)
plt.imshow(im_p100_ifft, plt.cm.gray)
plt.title('image 100')

plt.subplot(5, 4, 20)
plt.imshow(im_p10004_ifft, plt.cm.gray)
plt.title('image 10004')

plt.show()
