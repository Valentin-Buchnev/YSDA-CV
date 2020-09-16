import numpy as np


COLOR_MAP = {}


def save_result(fname, result):
    """Save tracking result to file."""
    with open(fname, 'w') as fout:
        for i, detection in enumerate(result):
            for line in detection:
                print(i, *line, sep=',', file=fout)


def load_result(fname):
    """Load tracking result from file"""
    with open(fname) as fin:
        result = [[int(x) for x in line.split(',')] for line in fin if len(line) > 0]
    result = np.array(result, dtype=np.int32)
    result = [result[result[:, 0] == i, 1:] for i in range(result[:, 0].max() + 1)]
    return result


def get_color(label):
    return COLOR_MAP.setdefault(label, np.random.randint(0, 256, size=3))
