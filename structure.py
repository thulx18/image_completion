import os
from PIL.Image import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from texture import texture_propagation

def plot(x):
    plt.imshow(x)
    plt.show()

def get_patch_mask(patch_point, patch_size, rows, cols):
    """获得邻域的像素"""
    row_left = max(0, patch_point[0] - (patch_size // 2))
    row_right = min(rows, patch_point[0] + (patch_size // 2) + 1)
    col_up = max(0, patch_point[1] - (patch_size // 2))
    col_down = min(cols, patch_point[1] + (patch_size // 2) + 1)
    patch_mask = np.zeros((rows, cols), dtype=np.bool)
    patch_mask[row_left:row_right, col_up:col_down] = True
    return patch_mask

def get_Es(source_point, target_point, unknown_curve, target_mask):
    normalize_by = np.bitwise_and(unknown_curve, target_mask).sum()
    return (np.linalg.norm(np.array(source_point) - np.array(target_point)) + np.linalg.norm(np.array(target_point) - np.array(source_point)) / normalize_by)

def get_Ei(source_patch_mask, source_patch, unknown_mask, target_patch_mask, img):
    Ei = 0.0
    coi_part = np.bitwise_and(source_patch_mask, unknown_mask)
    if coi_part.any():
        test_img = img.copy()
        test_img[target_patch_mask] = source_patch
        try:
            Ei = cv2.matchTemplate(img, source_patch, method=cv2.TM_SQDIFF_NORMED)[0][0]
        except Exception:
            Ei = 0.0
    return Ei

def get_E1(Es, Ei, ks = 50.0, ki = 2.0):
    return (ks * Es) + (ki * Ei)

def get_E2(source_patche1, source_patche2, target_patch_mask1, target_patch_mask2, image):
    E2 = 0.0
    coi_part = np.bitwise_and(target_patch_mask1, target_patch_mask2)
    if coi_part.any():
        t_img1 = np.zeros_like(image)
        t_img2 = np.zeros_like(image)
        try:
            t_img1[target_patch_mask1] = source_patche1
            t_img2[target_patch_mask2] = source_patche2
        except Exception:
            E2 = 0.0
        try:
            E2 = cv2.matchTemplate(t_img1[coi_part], t_img2[coi_part], method=cv2.TM_SQDIFF_NORMED)[0][0]
        except Exception:
            E2 = 0.0
    return E2

def structure_propagation(image_name, sampling_interval, patch_size, down_sample, to_show):
    print("begin to propagate structure")
    print(image_name, sampling_interval, patch_size, down_sample, to_show)
    # 读图
    image_path = os.path.join('image_data', image_name, image_name+'.png')
    structure_path = os.path.join('image_data', image_name, image_name+'_structure.png')
    mask_path = os.path.join('image_data', image_name, image_name+'_mask.png')
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    structure_image = (cv2.imread(structure_path, cv2.IMREAD_GRAYSCALE)).astype(np.bool)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.bool)
    if to_show:
        plot(image)
        plot(structure_image)
        plot(mask_image)
    # 下采样，加快
    image = image[::down_sample, ::down_sample, :]
    rows, cols, _ =  image.shape
    image_structure_prop  = image.copy()
    structure_image = structure_image[::down_sample, ::down_sample]
    mask_image = mask_image[::down_sample, ::down_sample]
    image_structure_prop[mask_image] = (0, 0, 0)
    contours = cv2.findContours(structure_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    updated_structure_mask = np.zeros_like(structure_image)
    structure_list = []
    for contour in contours:
        contour_mask = np.zeros(structure_image.shape, dtype=np.uint8)
        contour_image = cv2.fillPoly(contour_mask, pts=[contour], color=(255,255,255)).astype(np.bool)
        if to_show:
            plot(contour_image)
        structure_list.append(contour_image)
    
    for n, struc in enumerate(structure_list):
        print(n+1, "part of structure")
        # 生成锚点
        unknown_curve = np.bitwise_and(struc.astype(np.bool), mask_image.astype(np.bool))
        a_rows, a_cols = np.where(unknown_curve)
        a_r_c = tuple(zip(a_rows, a_cols))
        anchor_points = a_r_c[:: sampling_interval]
        # 选定需要补全的范围
        target_patch_masks = dict()
        for anchor_point in anchor_points:
            patch_mask = get_patch_mask(anchor_point, patch_size, rows, cols)
            target_patch_masks[anchor_point] = patch_mask
        for i in target_patch_masks:
            updated_structure_mask[target_patch_masks[i]] = True
        # 生成已知点
        known_curve = struc != unknown_curve
        p_rows, p_cols = np.where(known_curve)
        p_r_c = tuple(zip(p_rows, p_cols))
        patch_points = p_r_c[:: sampling_interval // 2]
        # 找到已知点中与未知部分无交集的
        source_patch_masks = dict()
        source_patches = dict()
        for patch_point in patch_points:
            patch_mask = get_patch_mask(patch_point, patch_size, rows, cols)
            if not np.bitwise_and(patch_mask, mask_image).any():
                source_patch_masks[patch_point] = patch_mask
        patch_points = tuple(list(source_patch_masks.keys()))
        for patch_point in patch_points:
            if patch_point in source_patch_masks:
                source_patches[patch_point] = image[source_patch_masks[patch_point]]
        
        # 计算最小能量差并补全
        energy = np.zeros((len(anchor_points), len(patch_points)), dtype=np.float64)
        for i in range(len(patch_points)):
            target_point = anchor_points[0]
            source_point = patch_points[i]
            Es = get_Es(source_point, target_point, unknown_curve, target_patch_masks[target_point])
            Ei = get_Ei(source_patch_masks[source_point], source_patches[source_point], mask_image, target_patch_masks[target_point], image)
            energy[0][i] = get_E1(Es, Ei)
        min_energy_index = np.ones(energy.shape) * np.inf
        for i in range(1, len(anchor_points)):
            for j in range(len(patch_points)):
                now_energy = np.inf
                now_index = np.inf
                source_point = patch_points[j]
                target_point = anchor_points[i] 
                Es = get_Es(source_point, target_point, unknown_curve, target_patch_masks[target_point])
                Ei = get_Ei(source_patch_masks[source_point], source_patches[source_point], mask_image, target_patch_masks[target_point], image)
                E1 = get_E1(Es, Ei)
                for k in range(len(patch_points)):
                    source_point1 = patch_points[j]
                    source_point2 = patch_points[k]
                    target_point1 = anchor_points[i - 1]
                    target_point2 = anchor_points[i] 
                    E2 = get_E2(source_patches[source_point1], source_patches[source_point2], target_patch_masks[target_point1], target_patch_masks[target_point2], image)
                    new_energy = energy[i - 1][k] + E2
                    if new_energy < now_energy:
                        now_energy = new_energy
                        now_index = k
                energy[i][j] = E1 + now_energy
                min_energy_index[i][j] = now_index

        # 寻找最佳替换点
        prop_points = list()
        t_prop_points = list()
        for i in range(energy.shape[0] - 1, -1, -1):
            idx = -1
            now_energy = np.inf
            for k in range(energy.shape[1]):
                new_energy = energy[i][k]
                if new_energy < now_energy:
                    now_energy = new_energy
                    idx = k
            node = min_energy_index[i][idx]
            prop_points.append(node)
        prop_points.reverse()
        prop_points = [int(patch) for patch in prop_points if np.isfinite(patch)]
        for patch_point in prop_points:
            if (source_patches[patch_points[patch_point]].size!= patch_size * patch_size * 3):
                node = patch_point - 1 if patch_point > 1 else patch_point + 1
                t_prop_points.append(node)
        if t_prop_points:
            prop_points = t_prop_points

        for i, patch_point in enumerate(prop_points):
            source_patch = source_patches[patch_points[int(patch_point)]]
            target_patch = target_patch_masks[anchor_points[i]]
            image_structure_prop[target_patch] = source_patch
    
    if to_show:
        plot(image_structure_prop)
    cv2.imwrite(os.path.join('image_data', image_name, image_name+'_stru.png'), image_structure_prop.astype(np.uint8))
    print("finish propagating structure")

    texture_propagation(image_name, image_structure_prop, updated_structure_mask, mask_image,to_show, image)