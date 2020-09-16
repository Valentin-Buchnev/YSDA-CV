# import numpy as np

# def get_img_by_shift(A, dx, dy):
#     return A[max(0, -dx):A.shape[0] - max(0, dx), max(0, -dy):A.shape[1] - max(0, dy)]

# def get_shift(A, B, s, step):
#     min_mse = ((A - B) ** 2).mean()
#     shift = (0, 0)
#     for dx in range(s[0] - step, s[0] + step + 1, step):
#         for dy in range(s[1] - step, s[1] + step + 1, step):
            
#             A_cur = get_img_by_shift(A, dx, dy)
#             B_cur = get_img_by_shift(B, -dx, -dy)
            
#             assert A_cur.shape == B_cur.shape
            
#             cur_mse = ((A_cur - B_cur) ** 2).mean()
            
#             if dx == 1 and dy == 1:
#                 print(A)
#                 print(B)
#                 print(cur_mse)
                    
#             if cur_mse < min_mse:
#                 min_mse = cur_mse
#                 shift = (dx, dy)
                
#     return shift

# def align(img, gg):
    
#     g_row, g_col = gg[0], gg[1]
    
#     h = img.shape[0] // 3
    
#     R = G = B = 0
    
#     if img.shape[0] % 3 != 0:
#         B, G, R, T = np.split(img, [h, 2 * h, 3 * h])
#     else:
#         B, G, R = np.split(img, 3)
        
#     w_del = R.shape[1] // 40
#     h_del = R.shape[0] // 40
    
#     R = R[h_del:R.shape[0] - h_del, w_del:R.shape[1] - w_del]
#     G = G[h_del:G.shape[0] - h_del, w_del:G.shape[1] - w_del]
#     B = B[h_del:B.shape[0] - h_del, w_del:B.shape[1] - w_del]
    
#     assert R.shape == G.shape
#     assert B.shape == G.shape
    
#     shift_R = shift_B = (0, 0)
    
#     for i in [64, 32, 16, 8, 4, 2, 1]:
#         shift_R = get_shift(G[::i], R[::i], shift_R, i) * 2
#         shift_B = get_shift(G[::i], B[::i], shift_B, i) * 2
        
#     h_res, w_res = get_img_by_shift(G, shift_R[0], shift_R[1]).shape
    
#     colored_img = np.zeros((h_res, w_res, 3)).astype(np.uint8)
#     r_row = g_row + h + shift_R[0]
#     r_col = g_col + shift_R[1]
#     b_row = g_row - h + shift_B[0]
#     b_col = g_col + shift_B[1]
    
#     return colored_img, (b_row, b_col), (r_row, r_col)

import numpy as np

def get_img_by_shift(A, dx, dy):
    return A[max(0, -dx):A.shape[0] - max(0, dx), max(0, -dy):A.shape[1] - max(0, dy)]

def get_shift(A, B, s, step):
    min_mse = ((A - B) ** 2).mean()
    shift = (0, 0)
    for dx in range(s[0] - step[0], s[0] + step[0]):
        for dy in range(s[1] - step[1], s[1] + step[1]):
            
            A_cur = get_img_by_shift(A, dx, dy)
            B_cur = get_img_by_shift(B, -dx, -dy)
            
            assert A_cur.shape == B_cur.shape
            
            cur_mse = ((A_cur - B_cur) ** 2).mean()
                    
            if cur_mse < min_mse:
                min_mse = cur_mse
                shift = (dx, dy)
                
    return np.asarray(shift)

def align(img, gg):
    
    input_img = img
    
    if img.shape[0] > 2000:
        img = img[::10, ::10]
    
    g_row, g_col = gg[0], gg[1]
    
    h = img.shape[0] // 3
    
    R = G = B = 0
    
    if img.shape[0] % 3 != 0:
        B, G, R, T = np.split(img, [h, 2 * h, 3 * h])
    else:
        B, G, R = np.split(img, 3)
        
    w_del = R.shape[1] // 10
    h_del = R.shape[0] // 10
    
    R = R[h_del:R.shape[0] - h_del, w_del:R.shape[1] - w_del]
    G = G[h_del:G.shape[0] - h_del, w_del:G.shape[1] - w_del]
    B = B[h_del:B.shape[0] - h_del, w_del:B.shape[1] - w_del]
    
    assert R.shape == G.shape
    assert B.shape == G.shape
    
    shift_R = get_shift(G, R, (0, 0), (10, 10))
    shift_B = get_shift(G, B, (0, 0), (10, 10))
    
    if input_img.shape[0] > 2000:
        img = input_img
        
        h = img.shape[0] // 3
    
        R = G = B = 0

        if img.shape[0] % 3 != 0:
            B, G, R, T = np.split(img, [h, 2 * h, 3 * h])
        else:
            B, G, R = np.split(img, 3)

        w_del = R.shape[1] // 10
        h_del = R.shape[0] // 10

        R = R[h_del:R.shape[0] - h_del, w_del:R.shape[1] - w_del]
        G = G[h_del:G.shape[0] - h_del, w_del:G.shape[1] - w_del]
        B = B[h_del:B.shape[0] - h_del, w_del:B.shape[1] - w_del]

        shift_R = get_shift(G, R, shift_R * np.asarray([10, 10]), (5, 5))
        shift_B = get_shift(G, B, shift_B * np.asarray([10, 10]), (5, 5))
    
    h_res, w_res = get_img_by_shift(G, shift_R[0], shift_R[1]).shape
    
    colored_img = np.zeros((h_res, w_res, 3)).astype(np.uint8)
    r_row = g_row + h + shift_R[0]
    r_col = g_col + shift_R[1]
    b_row = g_row - h + shift_B[0]
    b_col = g_col + shift_B[1]
    
    return colored_img, (b_row, b_col), (r_row, r_col)