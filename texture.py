import os
from PIL.Image import new
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve

height = 0
wight = 0

def plot(x):
    plt.imshow(x)
    plt.show()

def get_patch(patch_size, point):
    half_patch_size = (patch_size-1)//2
    patch = [
        [
            max(0, point[0] - half_patch_size),
            min(point[0] + half_patch_size, height-1)
        ],
        [
            max(0, point[1] - half_patch_size),
            min(point[1] + half_patch_size, width-1)
        ]
    ]
    return patch

def my_inpaint(image_stru, mask_image):
    patch_size = 10      #决定时间和效果
    image_stru = image_stru.astype('uint8')
    mask_image = mask_image.round().astype('uint8')
    out_img = image_stru.copy()
    global height, width
    height, width = image_stru.shape[:2]
    confidence = (1 - mask_image).astype(float)
    data = np.zeros([height, width])
    n = 0
    while True:
        if (n % 10 == 0):
            print(mask_image.sum(), "pixels left")
        n += 1
        # 确定边缘部分并更新信誉值
        edge = (laplace(mask_image) > 0).astype('uint8')
        temp_confidence = confidence.copy()
        edge_positions = np.argwhere(edge == 1)
        for point in edge_positions:
            patch = get_patch(patch_size, point)
            temp_confidence[point[0], point[1]] = sum(sum(
                confidence[patch[0][0]:patch[0][1]+1, patch[1][0]:patch[1][1]+1]
            )) / (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])
        confidence = temp_confidence
        
        # 计算data并更新优先级
        grey_image = rgb2gray(out_img)
        grey_image[mask_image == 1] = None
        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        temp_gradient = np.zeros([height, width, 2])
        for point in np.argwhere(edge == 1):
            patch = get_patch(patch_size,point)
            y_gradient = gradient[0][patch[0][0]:patch[0][1]+1, patch[1][0]:patch[1][1]+1]
            x_gradient = gradient[1][patch[0][0]:patch[0][1]+1, patch[1][0]:patch[1][1]+1]
            patch_gradient_val = gradient_val[patch[0][0]:patch[0][1]+1, patch[1][0]:patch[1][1]+1]
            patch_max_pos = np.unravel_index(patch_gradient_val.argmax(),patch_gradient_val.shape)
            temp_gradient[point[0], point[1], 0] = y_gradient[patch_max_pos]
            temp_gradient[point[0], point[1], 1] = x_gradient[patch_max_pos]
        gradient = temp_gradient

        x_n = convolve(mask_image.astype(float), np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]]))
        y_n = convolve(mask_image.astype(float), np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]]))
        normal = np.dstack((x_n, y_n))
        n_height, n_width = normal.shape[:2]
        norm = np.sqrt(y_n**2 + x_n**2).reshape(n_height, n_width, 1).repeat(2, axis=2)
        norm[norm == 0] = 1
        normal = normal / norm

        normal_gradient = normal*gradient
        data = np.sqrt(normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2) + 0.001
        priority = confidence * data * edge

        # 查找最佳匹配
        target_pixel = np.unravel_index(priority.argmax(),priority.shape)
        target_patch = get_patch(patch_size, target_pixel)
        patch_height, patch_width = (1+target_patch[0][1]-target_patch[0][0]), (1+target_patch[1][1]-target_patch[1][0])

        best_match = None
        temp_difference = 0
        lab_image = rgb2lab(out_img)
        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [[y, y + patch_height-1],[x, x + patch_width-1]]
                if mask_image[source_patch[0][0]:source_patch[0][1]+1, source_patch[1][0]:source_patch[1][1]+1].sum() != 0:
                    continue
                mask = 1 - mask_image[target_patch[0][0]:target_patch[0][1]+1, target_patch[1][0]:target_patch[1][1]+1]
                m_height, m_width = mask.shape
                rgb_mask = mask.reshape(m_height, m_width, 1).repeat(3, axis=2)
                target_data = (lab_image[target_patch[0][0]:target_patch[0][1]+1, target_patch[1][0]:target_patch[1][1]+1]) * rgb_mask
                source_data = (lab_image[source_patch[0][0]:source_patch[0][1]+1, source_patch[1][0]:source_patch[1][1]+1]) * rgb_mask
                squared_distance = ((target_data - source_data)**2).sum()
                euclidean_distance = np.sqrt(
                    (target_patch[0][0] - source_patch[0][0])**2 +
                    (target_patch[1][0] - source_patch[1][0])**2
                ) 
                difference = squared_distance + euclidean_distance
                if best_match is None or difference < temp_difference:
                    best_match = source_patch
                    temp_difference = difference
        
        # 更新图片
        source_patch = best_match
        pixels_positions = np.argwhere(
            mask_image[target_patch[0][0]:target_patch[0][1]+1, target_patch[1][0]:target_patch[1][1]+1] == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        target_confidence = confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            confidence[point[0], point[1]] = target_confidence
        mask = mask_image[target_patch[0][0]:target_patch[0][1]+1, target_patch[1][0]:target_patch[1][1]+1]
        m_height, m_width = mask.shape
        source_data = out_img[source_patch[0][0]:source_patch[0][1]+1, source_patch[1][0]:source_patch[1][1]+1]
        target_data = out_img[target_patch[0][0]:target_patch[0][1]+1, target_patch[1][0]:target_patch[1][1]+1]
        rgb_mask = mask.reshape(m_height, m_width, 1).repeat(3, axis=2)
        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)
        out_img[target_patch[0][0]:target_patch[0][1]+1,target_patch[1][0]:target_patch[1][1]+1] = new_data
        mask_image[target_patch[0][0]:target_patch[0][1]+1,target_patch[1][0]:target_patch[1][1]+1] = 0

        if mask_image.sum() == 0:
            break
    return out_img

def texture_propagation(image_name, image_stru, structure_image, mask_image, to_show, image):
    print("begin to propagate texture")
    new_mask_image = mask_image.copy()
    new_mask_image[mask_image == structure_image] = False
    text_mask = (cv2.cvtColor(image_stru, cv2.COLOR_BGR2GRAY) == 0).astype(np.uint8)
    # 使用cv2自带的补全函数作为效果对比
    image_texture_prop1 = cv2.inpaint(image_stru, text_mask, 3, cv2.INPAINT_NS)
    if to_show:
        plot(image_texture_prop1)
    cv2.imwrite(os.path.join('image_data', image_name, image_name+'_text1.png'), image_texture_prop1.astype(np.uint8))
    # 实现效果
    image_texture_prop2 = my_inpaint(image_stru, new_mask_image)
    cv2.imwrite(os.path.join('image_data', image_name, image_name+'_text2.png'), image_texture_prop2.astype(np.uint8))
    # 无结构信息
    image_no_stru_info = my_inpaint(image, mask_image)
    cv2.imwrite(os.path.join('image_data', image_name, image_name+'_no_stru_info.png'), image_no_stru_info.astype(np.uint8))
    print("finish propagating texture")