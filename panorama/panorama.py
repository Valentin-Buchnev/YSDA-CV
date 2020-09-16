import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=300):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """

    img2d = rgb2gray(img)
    
    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(img2d)
    kp = descriptor_extractor.keypoints
    des = descriptor_extractor.descriptors

    return kp, des


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))
    
    Cx = points.T[0].mean()
    Cy = points.T[1].mean()
    
    N = ((((pointsh[0] - Cx) ** 2) + ((pointsh[1] - Cy) ** 2)) ** 0.5).mean()
    
    N = (2 ** 0.5) / N
    
    matrix[0][0] = matrix[1][1] = N
    matrix[2][2] = 1
    matrix[0][2] = -N * Cx
    matrix[1][2] = -N * Cy
    
    return matrix, (matrix @ pointsh).T


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """
    
    n = src_keypoints.shape[0]

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    H = np.zeros((3, 3))

    A = []
    
    for i in range(n):
        A.append(np.asarray([-src[i][0], -src[i][1], -1, 0, 0, 0, 
                              dest[i][0] * src[i][0], dest[i][0] * src[i][1], dest[i][0]]))
        A.append(np.asarray([0, 0, 0, -src[i][0], -src[i][1], -1, 
                              dest[i][1] * src[i][0], dest[i][1] * src[i][1], dest[i][1]]))
    
    A = np.asarray(A)
    
    U, S, V = np.linalg.svd(A, full_matrices=True)
    
    H[0] = V[-1][:3]
    H[1] = V[-1][3:6]
    H[2] = V[-1][6:]
    
    H = inv(dest_matrix) @ H @ src_matrix

    return H


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=500, residual_threshold=2, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    
    inds_match = match_descriptors(src_descriptors, dest_descriptors)
    
    
    src_descriptors = src_descriptors[inds_match[:, 0]]
    dest_descriptors = dest_descriptors[inds_match[:, 1]]
    
    src_keypoints = src_keypoints[inds_match[:, 0]]
    dest_keypoints = dest_keypoints[inds_match[:, 1]]
    
    n = src_keypoints.shape[0]
    
    opt_cnt = -1
    good_inds = []
    
    for i in range(max_trials):
        inds = np.random.choice(n, 4, replace=False)

        H = find_homography(src_keypoints[inds], dest_keypoints[inds])

        H_dest_keypoints = (ProjectiveTransform(H)(src_keypoints))

        cur_cnt = (((dest_keypoints - H_dest_keypoints) ** 2).sum(axis=1) < residual_threshold).sum()

        if cur_cnt > opt_cnt:
            opt_cnt = cur_cnt
            good_inds = (((dest_keypoints - H_dest_keypoints) ** 2).sum(axis=1) < residual_threshold)
    
    res = ProjectiveTransform(find_homography(src_keypoints[good_inds], dest_keypoints[good_inds]))
    
    if return_matches:
        return res, inds_match[good_inds]
    return res


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()
    
    tr_m = []
    
    for i in range(image_count - 1):
        tr_m.append(forward_transforms[i].params)
    
    for i in range(center_index - 1, -1, -1):
        result[i] = result[i + 1] + ProjectiveTransform(tr_m[i])
    
    for i in range(center_index + 1, image_count):
        result[i] =  result[i - 1] + ProjectiveTransform(inv(tr_m[i - 1]))

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate([c for c in corners])
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    
    mn, mx = get_min_max_coords(get_corners(image_collection, simple_center_warps))
    
    final_center_warps = [scw for scw in simple_center_warps]
    
    shift = np.asarray([[1, 0, -mn[1]],
                        [0, 1, -mn[0]], 
                        [0, 0, 1]])
    
    for i in range(len(simple_center_warps)):
        final_center_warps[i] = ProjectiveTransform(shift @ final_center_warps[i].params)
    
    return tuple(final_center_warps), (int((mx - mn)[1]), int((mx - mn)[0]))


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    
    warped_image = warp(image, ProjectiveTransform(inv(rotate_transform_matrix(transform).params)), output_shape=output_shape)
    mask = np.zeros(output_shape, dtype=np.bool8)
    
    mask = (warped_image[:, :, 0] != 0) | (warped_image[:, :, 1] != 0) | (warped_image[:, :, 2] != 0)
    
    return warped_image, mask

def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros((output_shape[0], output_shape[1], 3))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    for i in range(len(image_collection)):
        cur_image, cur_mask = warp_image(image_collection[i], final_center_warps[i], output_shape)
        
        corners = tuple(get_corners((image_collection[i], ), (final_center_warps[i], )))
        
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                if cur_mask[i][j] and not result_mask[i][j]:
                    result_mask[i][j] = True
                    result[i, j] = cur_image[i, j]

    return result


def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    
    layers = [image]
    
    for i in range(n_layers - 1):
        image = gaussian(image, sigma)
        layers.append(image)
    
    return tuple(layers)


def get_laplacian_pyramid(image, n_layers=10, sigma=5):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    
    gp = get_gaussian_pyramid(image, n_layers, sigma)
    
    layers = []
    
    for i in range(n_layers - 1):
        layers.append(gp[i] - gp[i + 1])
    layers.append(gp[-1])
    
    return tuple(layers)


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=10, image_sigma=2, merge_sigma=1):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    
    n = len(image_collection)
    
    warped = []
    mask = []
    
    for i in range(n):
        w, m = warp_image(image_collection[i], final_center_warps[i], output_shape)
        warped.append(w)
        mask.append(m) 

    res = np.zeros(output_shape)
    
    for i in range(n - 1):
        cur_mask = mask[i] | mask[i + 1]
        and_mask = mask[i] & mask[i + 1]
        
        x_and_mask = (and_mask.max(axis=0) > 0)
        
        left = x_and_mask.argmax()
        right = x_and_mask.shape[0] - x_and_mask[::-1].argmax() - 1
        middle = (left + right) // 2
        
        cur_mask = np.hstack((cur_mask[:, :middle], np.zeros((output_shape[0], output_shape[1] - middle))))
        
        LA = get_laplacian_pyramid(warped[i], n_layers, image_sigma)
        LB = get_laplacian_pyramid(warped[i + 1], n_layers, image_sigma)
        GM_cur = get_gaussian_pyramid(cur_mask, n_layers, merge_sigma)
        
        LA = np.asarray([la for la in LA])
        LB = np.asarray([lb for lb in LB])
        GM_cur = np.asarray([gm for gm in GM_cur])
        
        LS = np.zeros(LA.shape)
        
        for k in range(3):
            LS[:, :, :, k] = GM_cur * LA[:, :, :, k] + (1 - GM_cur) * LB[:, :, :, k]
        
        
        
        warped[i + 1] = merge_laplacian_pyramid(tuple(LS))
        
        mask[i + 1] |= mask[i]

    return warped[-1]