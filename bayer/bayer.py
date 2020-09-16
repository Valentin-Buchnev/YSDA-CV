import numpy as np
import math

def get_bayer_masks(n_rows, n_cols):
    
    r, c = n_rows, n_cols
    if r % 2 == 1:
        r += 1
    if c % 2 == 1:
        c += 1
    
    R = np.tile([[0, 1], [0, 0]], (r // 2, c // 2))[:n_rows, :n_cols]
    G = np.tile([[1, 0], [0, 1]], (r // 2, c // 2))[:n_rows, :n_cols]
    B = np.tile([[0, 0], [1, 0]], (r // 2, c // 2))[:n_rows, :n_cols]

    return np.dstack((R, G, B))

def get_colored_img(raw_img):
    raw_img = np.array(raw_img)
    
    result = []
        
    masks = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    R = raw_img * masks[:, :, 0]
    G = raw_img * masks[:, :, 1]
    B = raw_img * masks[:, :, 2]
        
    return np.dstack((R, G, B))

def bilinear_interpolation(colored_img):
    colored_img = colored_img.astype(np.float64)

    masks = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])

    result = colored_img.copy()

    for i in range(colored_img.shape[0]):
        for j in range(colored_img.shape[1]):
            for k in range(colored_img.shape[2]):

                if masks[i][j][k] == 1:
                    continue

                if i == 0 or i == colored_img.shape[0] - 1:
                    continue
                if j == 0 or j == colored_img.shape[1] - 1:
                    continue

                res = 0
                cnt = 0

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if masks[i + dx][j + dy][k] == 0:
                            continue
                        res += colored_img[i + dx][j + dy][k]
                        cnt += 1
                result[i, j, k] = res / cnt

    return result.astype(np.uint8)
    
def improved_interpolation(raw_img):
    
    raw_img = get_colored_img(raw_img)
    raw_img = raw_img.astype(np.float64)
    
    result = raw_img
    
    masks = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    
    for i in range(raw_img.shape[0]):
            for j in range(raw_img.shape[1]):
                for k in range(raw_img.shape[2]):
    
                    if i <= 1 or i >= raw_img.shape[0] - 2:
                        continue
                    if j <= 1 or j >= raw_img.shape[1] - 2:
                        continue
                        
                    if masks[i][j][k] == 1:
                        result[i][j][k] = raw_img[i][j][k]
                        continue
                        
                    if k == 0:
                        if masks[i][j][1] and masks[i][j - 1][0]:
                            result[i][j][0] = raw_img[i][j][1] * 5
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][0] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][0] += raw_img[i][j + dy][1] * (-1)
                            
                            for dx in [-2, 2]:
                                result[i][j][0] += raw_img[i + dx][j][1] * 0.5
                            
                            for dy in [-1, 1]:
                                result[i][j][0] += raw_img[i][j + dy][0] * 4
                            
                            result[i][j][0] /= 8
                    
                        if masks[i][j][1] and masks[i - 1][j][0]:
                            result[i][j][0] = raw_img[i][j][1] * 5
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][0] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][0] += raw_img[i][j + dy][1] * 0.5
                            
                            for dx in [-2, 2]:
                                result[i][j][0] += raw_img[i + dx][j][1] * (-1)
                            
                            for dx in [-1, 1]:
                                result[i][j][0] += raw_img[i + dx][j][0] * 4
                            
                            result[i][j][0] /= 8
                            
                        if masks[i][j][2]:
                            result[i][j][0] = raw_img[i][j][2] * 6
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][0] += raw_img[i + dx][j + dy][0] * 2
                            
                            for d in [-2, 2]:
                                result[i][j][0] += raw_img[i + d][j][2] * (-1.5)
                                result[i][j][0] += raw_img[i][j + d][2] * (-1.5)
                                
                            result[i][j][0] /= 8
                            
                    if k == 1:
                        if masks[i][j][0]:
                            result[i][j][1] = raw_img[i][j][0] * 4
                            
                            for d in [-1, 1]:
                                result[i][j][1] += raw_img[i + d][j][1] * 2
                                result[i][j][1] += raw_img[i][j + d][1] * 2
                                
                            for d in [-2, 2]:
                                result[i][j][1] += raw_img[i + d][j][0] * (-1)
                                result[i][j][1] += raw_img[i][j + d][0] * (-1)
                        
                            result[i][j][1] /= 8 
                        
                        if masks[i][j][2]:
                            result[i][j][1] = raw_img[i][j][2] * 4
                            
                            for d in [-1, 1]:
                                result[i][j][1] += raw_img[i + d][j][1] * 2
                                result[i][j][1] += raw_img[i][j + d][1] * 2
                                
                            for d in [-2, 2]:
                                result[i][j][1] += raw_img[i + d][j][2] * (-1)
                                result[i][j][1] += raw_img[i][j + d][2] * (-1)
                        
                            result[i][j][1] /= 8 
                   
                    if k == 2:
                        if masks[i][j][1] and masks[i][j - 1][2]:
                            result[i][j][2] = raw_img[i][j][1] * 5
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][2] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][2] += raw_img[i][j + dy][1] * (-1)
                            
                            for dx in [-2, 2]:
                                result[i][j][2] += raw_img[i + dx][j][1] * 0.5
                            
                            for dy in [-1, 1]:
                                result[i][j][2] += raw_img[i][j + dy][2] * 4
                            
                            result[i][j][2] /= 8
                    
                        if masks[i][j][1] and masks[i - 1][j][2]:
                            result[i][j][2] = raw_img[i][j][1] * 5
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][2] += raw_img[i + dx][j + dy][1] * (-1)
                            
                            for dy in [-2, 2]:
                                result[i][j][2] += raw_img[i][j + dy][1] * 0.5
                            
                            for dx in [-2, 2]:
                                result[i][j][2] += raw_img[i + dx][j][1] * (-1)
                            
                            for dx in [-1, 1]:
                                result[i][j][2] += raw_img[i + dx][j][2] * 4
                            
                            result[i][j][2] /= 8
                            
                        if masks[i][j][0]:
                            result[i][j][2] = raw_img[i][j][0] * 6
                            
                            for dx in [-1, 1]:
                                for dy in [-1, 1]:
                                    result[i][j][2] += raw_img[i + dx][j + dy][2] * 2
                            
                            for d in [-2, 2]:
                                result[i][j][2] += raw_img[i + d][j][0] * (-1.5)
                                result[i][j][2] += raw_img[i][j + d][0] * (-1.5)
                                
                            result[i][j][2] /= 8
    return np.clip(result, 0, 255).astype(np.uint8)     

def compute_mse(img_pred, img_gt):
    return ((img_pred - img_gt) ** 2).mean()

def compute_psnr(img_pred, img_gt):
    
    mse = compute_mse(img_pred, img_gt)
    
    if mse == 0:
        raise ValueError
    
    return round(10 * math.log((img_gt.max() ** 2) / mse, 10), 14)