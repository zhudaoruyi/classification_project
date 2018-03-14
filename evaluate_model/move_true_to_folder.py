#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
from os.path import join
import shutil
from glob import glob
from tqdm import tqdm

base_dir = '/home/abc/pzw/data/class_2201/'

o_s = glob(join(base_dir, 'images', '*'))
print 'total num: %d ' % len(o_s)

for i, k in tqdm(enumerate(o_s)):
    with open(join(base_dir, 'labels', k.split('/')[-1] + '.txt'), 'r') as f:
        labels = f.read()
    true_label = labels.split('\n')
    while '' in true_label:
        true_label.remove('')
    true_label = sorted(true_label, reverse=True)
    # print true_label, len(true_label)

    dst_dirs = '/home/abc/pzw/evaluate_model/data/class_2201/'

    if not os.path.exists(dst_dirs):
        os.mkdir(dst_dirs)

    if true_label != [] and len(true_label) == 1:
        if not os.path.exists(dst_dirs + true_label[0]):
            os.mkdir(dst_dirs + true_label[0])
        shutil.copy(k, dst_dirs + true_label[0])
    if true_label != [] and len(true_label) == 2:
        if not os.path.exists(dst_dirs + str(true_label[0] + '_and_' + true_label[1])):
            os.mkdir(dst_dirs + str(true_label[0] + '_and_' + true_label[1]))
        shutil.copy(k, dst_dirs + str(true_label[0] + '_and_' + true_label[1]))
