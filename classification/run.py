#!/usr/bin/env python3

from json import dumps, load
from os import environ
from os.path import join
from sys import argv


def check_test(data_dir):
    with open(join(data_dir, 'output/output.csv')) as fout:
        lines = fout.readlines()
        output = {}
        for line in lines:
            filename, class_id = line.rstrip('\n').split(',')
            output[filename] = class_id

    with open(join(data_dir, 'gt/gt.csv')) as fgt:
        next(fgt)
        lines = fgt.readlines()
        gt = {}
        for line in lines:
            filename, class_id = line.rstrip('\n').split(',')
            gt[filename] = class_id

    correct = 0
    total = len(gt)
    for k, v in gt.items():
        if output[k] == v:
            correct += 1

    accuracy = correct / total

    res = 'Ok, accuracy %.4f' % accuracy
    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    result = results[-1]['status']
    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        accuracy_str = result[13:]
        accuracy = float(accuracy_str)

        if accuracy >= 0.93:
            mark = 10
        elif accuracy >= 0.90:
            mark = 8
        elif accuracy >= 0.85:
            mark = 6
        elif accuracy >= 0.80:
            mark = 4
        elif accuracy >= 0.75:
            mark = 2
        elif accuracy > 0:
            mark = 1
        else:
            mark = 0

        res = {'description': accuracy_str, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir):
    from fit_and_classify import fit_and_classify, extract_hog
    from glob import glob
    from numpy import zeros
    from os.path import basename, join
    from skimage.io import imread

    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    def read_gt(gt_dir):
        fgt = open(join(gt_dir, 'gt.csv'))
        next(fgt)
        lines = fgt.readlines()

        filenames = []
        labels = zeros(len(lines))
        for i, line in enumerate(lines):
            filename, label = line.rstrip('\n').split(',')[:2]
            filenames.append(filename)
            labels[i] = int(label)

        return filenames, labels

    def extract_features(path, filenames):
        hog_length = len(extract_hog(imread(join(path, filenames[0]))))
        data = zeros((len(filenames), hog_length), dtype='float32')
        for i in range(0, len(filenames)):
            filename = join(path, filenames[i])
            data[i, :] = extract_hog(imread(filename))
        return data

    train_filenames, train_labels = read_gt(train_dir)
    test_filenames = []
    for path in sorted(glob(join(test_dir, '*png'))):
        test_filenames.append(basename(path))

    train_features = extract_features(train_dir, train_filenames)
    test_features = extract_features(test_dir, test_filenames)

    y = fit_and_classify(train_features, train_labels, test_features)

    with open(join(output_dir, 'output.csv'), 'w') as fout:
        for i, filename in enumerate(test_filenames):
            print('%s,%d' % (filename, y[i]), file=fout)


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print('Usage: %s mode data_dir output_dir' % argv[0])
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
            print('Usage: %s tests_dir' % argv[0])
            exit(0)

        from glob import glob
        from json import dump
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from os.path import exists
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

            test_num = input_dir[-8:-6]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                print(test_num, '%.2fs' % running_time, status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
