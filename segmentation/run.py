#!/usr/bin/env python3

from glob import glob
from json import load, dump, dumps
from os import environ
from os.path import basename, join, exists, splitext
from sys import argv
import numpy as np
from skimage.io import imread
from itertools import groupby

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rle_encode(img, filename):
    # img must be binary
    # encoding format:
    # [height, width, number of zeros, number of ones, number of zeros...]
    # (the first 'number of zeros' can be zero if img[0][0] = 1)
    h, w = img.shape[:2]
    data = [h, w]
    img = np.reshape(img, -1)
    for k, g in groupby(img):
        if len(data) == 2 and k == 1:  # img[0][0] = 1
            data.append(0)
        data.append(len(list(g)))
    with open(filename, 'w') as f:
        dump(data, f)


def rle_decode(filename):
    with open(filename, 'r') as f:
        data = load(f)
    h, w = data[:2]
    img = []
    for i in range(2, len(data)):
        img.extend([i % 2 for k in range(data[i])])
    img = np.reshape(img, (h, w))
    return img


def get_iou(l_true, l_pred):
    if np.max(l_true) > 1:
        l_true = l_true / 255
    if np.max(l_pred) > 1:
        l_pred = l_pred / 255
    return np.sum(np.clip(l_true * l_pred, 0, 1)) / (np.sum(np.clip(l_true + l_pred,  0, 1)))


def check_test(data_dir):
    gt_dir = join(data_dir, 'gt')
    output_dir = join(data_dir, 'output')

    filenames = glob(join(gt_dir, '**/*.png'))

    res_iou = 0
    all_found = True
    for filename in filenames:
        name, ext = splitext(basename(filename))

        out_filename = join(output_dir, name + '.txt')
        if not exists(out_filename):
            res = 'Error, segmentation for "{name}" not found'
            all_found = False
            res_iou = 0
            break

        pred_segm = rle_decode(out_filename)
        gt_segm = imread(filename, as_gray=True)
        iou = get_iou(gt_segm, pred_segm)
        res_iou += iou

    res_iou /= len(filenames)
    if all_found:
        res = f'Ok, IoU {res_iou:.4f}'

    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    result = results[-1]['status']

    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        iou_str = result[8:]
        iou = float(iou_str)

        if iou >= 0.80:
            mark = 10
        elif iou >= 0.75:
            mark = 9
        elif iou >= 0.70:
            mark = 8
        elif iou >= 0.65:
            mark = 7
        elif iou >= 0.60:
            mark = 6
        elif iou >= 0.50:
            mark = 5
        elif iou >= 0.40:
            mark = 4
        elif iou >= 0.30:
            mark = 3
        elif iou >= 0.2:
            mark = 2
        else:
            mark = 0

        res = {'description': iou_str, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir):
    from segmentation import predict
    from tensorflow.keras.models import load_model
    from os.path import abspath, basename, dirname, join

    code_dir = dirname(abspath(__file__))

    # from segmentation import train_segmentation_model
    # model = train_segmentation_model(join(data_dir, 'train'))

    model = load_model(join(code_dir, 'segmentation_model.hdf5'))

    img_filenames = glob(join(data_dir, 'test/images/**/*.jpg'))

    for filename in img_filenames:
        segm = (predict(model, filename) > 0.5).astype('uint8')
        name, ext = splitext(basename(filename))
        rle_encode(segm, join(output_dir, name + '.txt'))


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print(f'Usage: {argv[0]} tests_dir')
            exit(0)

        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_*_input'))):
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = join(output_dir, 'output')
            makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = join(output_dir, 'gt')
            if not exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time()
                run_single_test(input_dir, run_output_dir)
                end = time()
                running_time = end - start
            except:
                status = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except:
                    status = 'Checker error'
                    traceback = format_exc()

            test_num = basename(input_dir)[:2]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                print(test_num, f'{running_time:.2f}s', status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
