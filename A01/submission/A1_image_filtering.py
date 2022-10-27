import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

LENNA_PATH = "lenna.png"
SHAPE_PATH = "shapes.png"

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
    post_hight = img_height - ker_width + 1
    row_pad = kernel.shape[0] // 2
    col_pad = kernel.shape[1] // 2

    filtered = np.zeros(img.shape)
    for i in range(row_pad, row_pad + post_hight) :
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

def nine_gausians(
    img_name,
    filter_params = [
        [2, 1],
        [2, 5],
        [2, 10],
        [8, 1],
        [8, 5],
        [8, 10],
        [11, 1],
        [11, 5],
        [11, 10]
    ]
) :
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    imgs = list(map(
        lambda filter_param : cross_correlation_2d(
            img, get_gaussian_filter_2d(*filter_param)
        ),
        filter_params
    ))


    cv2.imshow(WINDOW_NAME, imgs[0])
    cv2.waitKey(-1)

    imgs = list(map(
        lambda img, filter_param : cv2.putText(
            img,
            "{0}x{0}, s={1}".format(*filter_param),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            2
        ),
        imgs,
        filter_params
    ))

    """
    result = np.vstack(
        [
            np.hstack([
                imgs[i + j] for j in range(3)
            ]) 
        ] for i in range(0, len(imgs), 3)
    )
    """
    return np.vstack([
        np.hstack([imgs[0], imgs[1], imgs[2]]),
        np.hstack([imgs[3], imgs[4], imgs[5]]),
        np.hstack([imgs[6], imgs[7], imgs[8]]),
    ])

def do_e(
    img_name,
    filter_param = [3, 1]
) :
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    kernel = get_gaussian_filter_2d(*filter_param)
    kernel_v = get_gaussian_filter_1d(*filter_param)
    kernel_h = get_gaussian_filter_1d(*filter_param)

    start_2d = time.time()
    filtered_2d = cross_correlation_2d(img, kernel)
    time_2d = time.time() - start_2d
    print("time consumed for 2D kernel filtering :", time_2d)

    start_1ds = time.time()
    filtered_1ds = cross_correlation_2d(
        cross_correlation_2d(img, kernel_v),
        kernel_h
    )
    time_1ds = time.time() - start_1ds
    print("time consumed for sequential 1D kernel filtering :", time_1ds)

    diff_map = np.abs(filtered_2d - filtered_1ds)

    cv2.imshow(WINDOW_NAME, diff_map)
    key = cv2.waitKey(-1)

def do_work(img_name) :
    print("image name :",img_name)
    print("applying 9 gaussian filteres...")


    nine_image = nine_gausians(img_name)
    cv2.imshow(WINDOW_NAME, nine_image)
    key = cv2.waitKey(-1)
    cv2.imwrite(f"./result/part_1_gaussian_filtered_{img_name}", nine_image)
    print("applying 1d, 2d gausian filters..")
    do_e(img_name)

    print()

if __name__ == "__main__" :
    print("gaussian_filter_1d(5, 1) :")
    print(get_gaussian_filter_1d(5, 1))
    print("gaussian_filter_2d(5, 1) :")
    print(get_gaussian_filter_2d(5, 1))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 512 * 3, 512 * 3)

    img_names = [LENNA_PATH, SHAPE_PATH]

    for img_name in img_names :
        do_work(img_name)