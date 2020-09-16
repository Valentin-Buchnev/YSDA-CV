from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Softmax, Conv2D
from tensorflow.keras.optimizers import Adam

import numpy as np

# ============================== 1 Classifier model ============================

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(2))
    model.add(Softmax())
    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))
    model.compile(Adam(), loss='binary_crossentropy')
    model.fit(X, y, batch_size=64, epochs=200)
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    model = Sequential()
    model.add(Conv2D(filters=2, kernel_size=(40, 100), input_shape=(None, None, 1)))
    model.add(Softmax(axis=-1))
    model.get_layer('conv2d').set_weights([
        cls_model.get_layer('dense').get_weights()[0].reshape(model.get_layer('conv2d').get_weights()[0].shape),
        cls_model.get_layer('dense').get_weights()[1]
    ])

    return model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images, threshold=0.95):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    names, images = zip(*dictionary_of_images.items())
    images = list(images)
    original_shapes = [None] * len(images)

    for i in range(len(images)):

        original_shapes[i] = images[i].shape

        images[i] = np.pad(images[i], ((0, max(0, 220 - images[i].shape[0])), (0, max(0, 370 - images[i].shape[1]))))
        images[i] = images[i][..., None]

    images = np.asarray(images)
    heatmaps = detection_model.predict(images)[..., 1]

    detections = {}
    for name, heatmap, original_shape in zip(names, heatmaps, original_shapes):

        shrinked_heatmap = heatmap[:(original_shape[0] - 40 + 1), :(original_shape[1] - 100 + 1)]
        candidates_args = np.argwhere(shrinked_heatmap >= threshold)

        detections[name] = np.hstack((
            candidates_args,
            40 * np.ones((len(candidates_args), 1)),
            100 * np.ones((len(candidates_args), 1)),
            shrinked_heatmap[shrinked_heatmap >= threshold].reshape(len(candidates_args), 1)
        ))

    return detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first_bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    r1, c1, h1, w1 = first_bbox
    r2, c2, h2, w2 = second_bbox

    nr = max(0, min(r1 + h1 - 1, r2 + h2 - 1) - max(r1, r2) + 1)
    nc = max(0, min(c1 + w1 - 1, c2 + w2 - 1) - max(c1, c2) + 1)

    S = h1 * w1 + h2 * w2 - nr * nc

    return nr * nc / S
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    tp = []
    fp = []

    cnt_gt = 0

    for name in pred_bboxes.keys():

        cur_gt = gt_bboxes[name]
        cnt_gt += len(cur_gt)
        used = [False] * len(cur_gt)

        cur_pred = pred_bboxes[name]
        cur_pred = sorted(cur_pred, key=lambda x: x[-1], reverse=True)

        for bbox in cur_pred:
            iou_idx = sorted(range(len(cur_gt)), key=lambda x: calc_iou(bbox[:4], cur_gt[x]), reverse=True)

            is_tp = False
            for idx in iou_idx:
                if not used[idx] and calc_iou(bbox[:4], cur_gt[idx]) >= 0.5:
                    is_tp = True
                    used[idx] = True
                    tp.append(bbox[-1])
                    break

            if not is_tp:
                fp.append(bbox[-1])

    tp = sorted(tp)
    tp_fp = sorted(tp + fp)
    rpc = []

    idx = 0
    for i, x in enumerate(tp_fp):
        while idx < len(tp) and tp[idx] < x:
            idx += 1
        cnt_tp = len(tp) - idx
        cnt_tp_fp = len(tp_fp) - i
        rpc.append((cnt_tp / cnt_gt, cnt_tp / cnt_tp_fp, x))

    rpc.append((0, 1, 1))
    rpc = np.array(rpc)
    return np.sum(((rpc[:-1, 1] + rpc[1:, 1]) * (rpc[:-1, 0] - rpc[1:, 0])) / 2)
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.25):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    result = dict()
    for name in detections_dictionary.keys():

        det = sorted(detections_dictionary[name], key=lambda x: x[-1], reverse=True)
        removed_det = [False] * len(det)

        for i in range(len(det)):
            if not removed_det[i]:
                for j in range(i + 1, len(det)):
                    if calc_iou(det[i][:-1], det[j][:-1]) >= iou_thr:
                        removed_det[j] = True

        other_idx = list(filter(lambda x: not removed_det[x], range(len(removed_det))))
        other_det = [det[x] for x in other_idx]

        result[name] = other_det

    return result
    # your code here /\
