import time
from tkinter import image_names
import numpy as np
import cv2

LENNA_PATH = "../images/lenna.png"
SHAPE_PATH = "../images/shapes.png"

WINDOW_NAME = "vis_window"

def get_gaussian_filter_1d(size, sigma) :
    kernel = np.arange(size//2 * -1, size//2+1).astype(np.float32).reshape(1, -1)
    kernel = np.exp( -kernel*kernel / (2*sigma*sigma) )
    kernel /= kernel.sum()
    return kernel

def get_gaussian_filter_2d(size, sigma) :
    x_grid, y_grid = np.meshgrid(
        np.arange(-2, 3),
        np.arange(-2, 3)
    )
    distance_square_map = x_grid ** 2 + y_grid ** 2
    kernel = np.exp(-distance_square_map / (2*sigma*sigma))
    kernel /= kernel.sum()
    return kernel


def get_gaussian_filter_2d_using_1d(size, sigma) :
    """
    넘파이 outer 연산을 써도 되는지 안되는지 모르겠어서 이 함수는 안 쓰기로 함
    """
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    kernel = np.outer(kernel_1d, kernel_1d)
    kernel /= kernel.sum()
    return kernel

def cross_correlation_1d(img, kernel) :
    """
    distinguish vertical, horizontal kernel based on the shape of the given kernel
    assume odd sized kernel
    preserve image shape
    
    assume both img and kernel are 1D
    

    """
    img_ = img.reshape(-1)
    kernel_ = kernel.reshape(-1)
    #  vertically stack images
    img_appended = np.vstack(list(map(
        lambda len_shift : img_[
            len_shift : len_shift + len(img_) - len(kernel_) + 1
        ],
        range(len(kernel_))
    )))
    filtered_ = np.matmul(kernel.reshape(1, -1), img_appended).reshape(-1)
    filtered = np.zeros(len(img_)) #, dtype=np.uint8)
    
    # pad  filtered  first
    pad_size = len(kernel_) // 2
    filtered[list(range(pad_size))] = filtered_[0]
    filtered[list(range(pad_size, len(filtered)))] = filtered_[-1]
    # put filtered_  to  filtered
    filtered[pad_size:-pad_size] = filtered_
    
    # check direction of kernel
    if len(kernel.shape) == 1 : # 1D
        filtered = filtered.reshape(1, -1)
    elif len(kernel.shape) == 2 : # 2D.  dimensions over 2D is not considered.
        if kernel.shape[0] > kernel.shape[1] :
            filtered = filtered.reshape(-1, 1)
        else :
            filtered = filtered.reshape(1, -1)
    else :
        pass
    return filtered

def cross_corr_2d_1(img, kernel) :
    img_height, img_width = img.shape
    ker_height, ker_width = kernel.shape
    post_width = img_width - ker_width + 1
    post_hight = img_height - ker_height + 1
    row_pad = kernel.shape[0] // 2
    col_pad = kernel.shape[1] // 2

    #print(kernel.shape, row_pad, col_pad, img_height, img_width, post_hight, post_width)

    filtered = np.zeros(img.shape)
    for i in range(row_pad, img_height - row_pad) :
        for j in range(col_pad, col_pad + post_width) :
            filtered[i][j] = np.sum(
                img[i-row_pad : i+row_pad+1 ,  j-col_pad : j+col_pad+1 ] * kernel
            )
    
    filtered[ list(range(row_pad)),:] = filtered[row_pad,:]
    filtered[
        list(range(img_height - row_pad, img_height)), 
        :
    ] = filtered[row_pad + post_hight - 1,:]
    
    filtered[:, list(range(col_pad)) ] = filtered[:,col_pad].reshape(-1, 1)
    filtered[
        :,
        list(range(img_width - col_pad, img_width))
    ] = filtered[:, img_width - col_pad - 1].reshape(-1, 1)

    return filtered

def cross_correlation_2d(img, kernel) :
    """
    """
    return cross_corr_2d_1(img, kernel)   

def compute_image_gradient(img) :
    sobel_h = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_v = sobel_h.copy().T

    grad_h = cross_correlation_2d(img, sobel_h)
    grad_v = cross_correlation_2d(img, sobel_v)

    mag = np.hypot(grad_h, grad_v)
    mag = mag / mag.max() * 255
    dir = np.arctan2(grad_v, grad_h)
    return mag, dir

def non_maximum_suppression_dir(mag, dir) :
    img_height, img_width = mag.shape
    suppressed = np.zeros(mag.shape)
    for i in range(1,img_height-1):
        for j in range(1,img_width-1):
            front = 255
            back = 255
            # 45, 225
            if (np.deg2rad(22.5) <= dir[i,j] < np.deg2rad(67.5)) :
                front = mag[i+1, j-1]
                back = mag[i-1, j+1]
            # 90, 270
            elif (np.deg2rad(67.5) <= dir[i,j] < np.deg2rad(112.5)):
                front = mag[i+1, j]
                back = mag[i-1, j]
            # 135, 315
            elif (np.deg2rad(112.5) <= dir[i,j] < np.deg2rad(157.5)):
                front = mag[i-1, j-1]
                back = mag[i+1, j+1]
            # 0, 180
            else : 
                front = mag[i, j+1]
                back = mag[i, j-1]

            if mag[i,j] >= front and mag[i,j] >= back:
                suppressed[i,j] = mag[i,j]
            else:
                suppressed[i,j] = 0
    return suppressed


def doPart2(
    img_name
) :
    image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    gaussian_filter = get_gaussian_filter_2d(7, 1.5)

    print("computing gradient of ", img_name)
    start_time = time.time()
    mag, dir = compute_image_gradient(image)
    print("took", time.time() - start_time, "second")

    cv2.imshow(WINDOW_NAME, mag.astype(np.uint8))
    cv2.waitKey(-1)
    #cv2.imwrite(f"./result/part_2_edge_raw_{image_names}", mag) 

    print("non maximum suppressing", img_name, "...")
    start_time = time.time()
    suppressed = non_maximum_suppression_dir(mag, dir)
    print("took", time.time() - start_time, "second")

    cv2.imshow(WINDOW_NAME, suppressed.astype(np.uint8))
    cv2.waitKey(-1)
    #cv2.imwrite(f"/result/part_2_edge_sup_{img_name}", suppressed.astype(np.uint8))


if __name__ == "__main__" :
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    img_names = [LENNA_PATH, SHAPE_PATH]

    for img_name in img_names :
        doPart2(img_name)