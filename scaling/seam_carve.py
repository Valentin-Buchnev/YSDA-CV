import numpy as np

def get_grad(Y, mask):
    I = np.zeros(Y.shape)
    
    for i in range(I.shape[0]):
        if i == 0:
            I[i] = Y[i + 1] - Y[i]
        elif i == I.shape[0] - 1:
            I[i] = Y[i] - Y[i - 1]
        else:
            I[i] = Y[i + 1] - Y[i - 1]
    
    return I

def get_mask(I, direction):
    
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if i > 0:
                add = I[i - 1][j]
                if j > 0:
                    add = min(add, I[i - 1][j - 1])
                if j < I.shape[1] - 1:
                    add = min(add, I[i - 1][j + 1])
                I[i][j] += add
    
    mask = np.zeros(I.shape)
    
    cur_id = 0
    
    for  j in range(0, I.shape[1]):
        if I[-1, j] < I[-1, cur_id]:
            cur_id = j
    
    for i in range(I.shape[0] - 1, 0, -1):
        mask[i][cur_id] = 1
        new_cur_id = max(cur_id - 1, 0)
        for j in range(max(cur_id - 1, 0), min(I.shape[1] - 1, cur_id + 1) + 1):
                if I[i - 1][j] < I[i - 1][new_cur_id]:
                        new_cur_id = j
        cur_id = new_cur_id
    
    mask[0][cur_id] = 1
    
    return mask

def delete_mask(img, mask):
    
    img_res = np.zeros((img.shape[0], img.shape[1] - 1, 3))
    
    for k in range(3):
        for i in range(img.shape[0]):
            index = mask[i].argmax()
            img_res[i,:,k] = np.hstack((img[i, :index, k], img[i, index + 1:, k]))
    
    return img_res

def seam_carve(img, work_type, mask=None):
    
    if mask is None:
        mask = np.zeros((img.shape[0], img.shape[1]))
        
    img = img.astype(np.float64)
    mask = mask.astype(np.float64)
    
    work_type = work_type.split(' ')
    
    if work_type[0] == 'vertical':
        img = img.transpose((1, 0, 2))
        mask = mask.T
    
    Y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    
    I_x = get_grad(Y, mask)
    I_y = get_grad(Y.T, mask.T).T
    
    I = (I_x ** 2 + I_y ** 2) ** 0.5
    
    I += mask * I.shape[0] * I.shape[1] * 256
    
    mask = get_mask(I, work_type[0])
    
    img = delete_mask(img, mask)
    
    if work_type[0] == 'vertical':
        img = img.transpose((1, 0, 2))
        mask = mask.T
    
    return img, None, mask