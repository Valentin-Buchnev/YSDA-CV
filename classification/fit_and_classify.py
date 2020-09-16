import numpy as np
import math
from skimage.transform import resize
from scipy.signal import convolve2d
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def extract_hog(img):

    cell_rows = 4
    cell_cols = 4
    block_row_cells = 5
    block_col_cells = 5
    bin_count = 10
    
    img = resize(img, (32, 32))
    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    S_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    S_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    I_x = convolve2d(img, S_x, mode='same', boundary='symm')
    I_y = convolve2d(img, S_y, mode='same', boundary='symm')
    G = (I_x ** 2 + I_y ** 2) ** 0.5
    T = np.arctan2(I_y, I_x)

    T = T / math.pi * 180. + 180
    T[T >= 360] = 0
    T = np.floor(T).astype(np.int64)

    bins = []
    
    for i in range(0, img.shape[0], cell_rows):
        
        cur = []
        
        for j in range(0, img.shape[1], cell_cols):
            
            T_cur = T[i:i + cell_rows, j:j + cell_cols]
            G_cur = G[i:i + cell_rows, j:j + cell_cols]

            v = np.zeros(bin_count)
            
            for x in range(cell_rows):
                for y in range(cell_cols):
                    v[T_cur[x][y] // (360 // bin_count)] += G_cur[x][y]

            cur.append(v)
            
        bins.append(cur)

    bins = np.asarray(bins)
    
    hog = np.array([])

    for i in range(0, bins.shape[0], block_row_cells):
        for j in range(0, bins.shape[1], block_col_cells):
            bins_cur = bins[i:i + block_row_cells, j:j + block_col_cells].flatten()
            hog = np.append(hog, bins_cur / np.sqrt((bins_cur ** 2).sum() + 1e-7))

    return hog


def fit_and_classify(X_train, y_train, X_test):
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(penalty='l2', C=0.01, max_iter=1000))
    ])

    clf.fit(X_train, y_train)
    
    return clf.predict(X_test)
