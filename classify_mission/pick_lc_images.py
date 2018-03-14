#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
import datetime
from tqdm import tqdm


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def list_images(path, file_type='images'):
    """
    列出文件夹中所有的文件，返回
    :param file_type: 'images' or 'any'
    :param path: a directory path, like '../data/pics'
    :return: all the images in the directory
    """
    image_suffix = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif']
    paths = []
    for file_and_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_and_dir)):
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in image_suffix:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
            elif file_type == 'any':
                paths.append(os.path.abspath(os.path.join(path, file_and_dir)))
            else:
                if os.path.splitext(file_and_dir)[1] == file_type:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
    return paths


if __name__ == '__main__':

    # 需要分类的数据集
    date_suffix = '0211'
    test_dir = '/home/abc/pzw/classify_mission/data/data' + date_suffix
    test_set = list_images(test_dir)
    print '数据集的图片数量是:', len(test_set), '\n'

    confidence_dict = json.loads(open('results/json/data' + date_suffix + '_lf.json').read(), encoding='utf-8')
    confidence_dict_sorted = sorted(confidence_dict.items(), key=lambda x: x[1])

    pr = 0
    print '\n', 'picture 95th-105th :', confidence_dict_sorted[95:105], '\n'
    print '\n', 'picture 495th-505th :', confidence_dict_sorted[495:505], '\n'
    print '\n', 'picture 995th-1005th :', confidence_dict_sorted[995:1005], '\n'
    if pr:
        print '\n', 'picture 1995th-2005th :', confidence_dict_sorted[1995:2005], '\n'
        print '\n', 'picture 2995th-3005th :', confidence_dict_sorted[2995:3005], '\n'
        print '\n', 'picture 3995th-4005th :', confidence_dict_sorted[3995:4005], '\n'
        print '\n', 'picture 4995th-5005th :', confidence_dict_sorted[4995:5005], '\n'
        print '\n', 'picture 5995th-6005th :', confidence_dict_sorted[5995:6005], '\n'

    if not pr:
        print 'current time is:', current_time()
        for i in tqdm(range(52548)):
            dst = 'pickouts/results' + date_suffix + '_ct'
            if not os.path.exists(dst):
                os.makedirs(dst)
            shutil.copy(test_dir + '/' + confidence_dict_sorted[i+3000][0], dst)
        print 'current time is:', current_time()


