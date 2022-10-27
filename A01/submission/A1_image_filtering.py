import numpy as np
import cv2


LENNA_PATH = "lenna.png"
SHAPE_PATH = "shapes.png"

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

def g2ggg(img) :
    return np.stack((img,img, img))


def doAsProffesorSaid(img_name) :




    list(map(

    ))



if __name__ == "__main__" :
    print("gaussian_filter_1d(5, 1) :")
    print(get_gaussian_filter_1d(5, 1))
    print("gaussian_filter_2d(5, 1) :")
    print(get_gaussian_filter_2d(5, 1))

    lenna_img = cv2.imread(LENNA_PATH, cv2.IMREAD_GRAYSCALE)
    shape_img = cv2.imread(SHAPE_PATH, cv2.IMREAD_GRAYSCALE)


    img_name = f"./result/part_1_gaussian_filtered_{}"