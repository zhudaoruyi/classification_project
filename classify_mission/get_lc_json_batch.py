#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import cv2
import time
import json
import shutil
import datetime
import numpy as np
from tqdm import tqdm
#from keras.models import load_model

# 单类 18
classes = ['LINE_CHART', 'AREA_CHART', 'BAR_CHART',
           'COLUMN_CHART', 'PIE_CHART', 'UNKNOWN',
           'GRID_TABLE', 'LINE_TABLE', 'QR_CODE',
           'INFO_GRAPH', 'TEXT', 'CANDLESTICK_CHART',
           'PHOTOS', 'SCATTER_CHART', 'RADAR_CHART',
           'DONUT_CHART', 'LINE_POINT_CHART', 'DISCRETE_PLOT']
chart_type_dic = {
    0: 'LINE_CHART',
    1: 'AREA_CHART',
    2: 'BAR_CHART',
    3: 'COLUMN_CHART',
    4: 'PIE_CHART',
    5: 'UNKNOWN',
    6: 'GRID_TABLE',
    7: 'LINE_TABLE',
    8: 'QR_CODE',
    9: 'INFO_GRAPH',
    10: 'TEXT',
    11: 'CANDLESTICK_CHART',
    12: 'PHOTOS',
    13: 'SCATTER_CHART',
    14: 'RADAR_CHART',
    15: 'DONUT_CHART',
    16: 'LINE_POINT_CHART',
    17: 'DISCRETE_PLOT'
}

# 导入模型
#model = load_model('/home/abc/pzw/files/class_17/train0205/models_and_logs/m5016_ir_01_p.h5')
#print 'testing model:', model.predict(np.zeros((1, 299, 299, 3)))

# 需要分类的数据集
test_dir = 'data/data0211'


def list_images(path, file_type='images'):
    """
    列出文件夹中所有的文件，返回
    :param file_type: 'images' or 'any'
    :param path: a directory path, like '../data/pics'
    :return: all the images in the directory
    """
    IMAGE_SUFFIX = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif']
    paths = []
    for file_and_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_and_dir)):
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in IMAGE_SUFFIX:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
            elif file_type == 'any':
                paths.append(os.path.abspath(os.path.join(path, file_and_dir)))
            else:
                if os.path.splitext(file_and_dir)[1] == file_type:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
    return paths


def low_confidence(pred):
    pred = pred - 0.5
    pred = np.abs(pred)
    confidence = np.min(pred)
    return confidence


test_set = sorted(list_images(test_dir))
print '需要分类的图片总数是：%d' % len(test_set)


def del_unopen(test_set, ft=False):
    print '删除打不开的图片...'
    for i, k in tqdm(enumerate(test_set)):
        if ft:
            print k.split('/')[-1]
        if cv2.imread(k) is None:
            del(test_set[i])
            os.remove(k)
    print '现在需要分类的图片总数是：%d' % len(test_set)


#del_unopen(test_set)
#del_unopen(test_set)


def predict(model, batch_size=64, test_set=test_set):
    """
    本函数调用model.predict进行预测，把所有的被预测对象写入x中，
    传给model去predict,速度很快，达到120it/s
    """

    x, low_conf = [], []
    for i, k in tqdm(enumerate(test_set)):
        img = cv2.imread(k)[:, :, ::-1]
        img = cv2.resize(img, (299, 299))
        x.append(np.array(img)/255.)
    x = np.array(x)
    print 'shape of x:', x.shape
    print 'Let us start predicting...'
    start = time.clock()
    predictions = model.predict(x, batch_size=batch_size)
    print 'time spending:', time.clock() - start
    for pred in predictions:
        confidence = low_confidence(pred[0])
        low_conf.append(confidence.tolist())
    return low_conf


def predict_on_batch(model, batch_size=64, test_set=test_set):
    """
    本函数调用model.predict_on_batch预测，按批次传入x，速度85it/s
    """

    low_conf = []
    for start in tqdm(range(0, len(test_set), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(test_set))
        new_batch = test_set[start:end]
        for jet in new_batch:
            img = cv2.imread(jet)[:, :, ::-1]
#            img = cv2.resize(img, (299, 299))
            img = cv2.resize(img, (224, 224))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255.
        predictions = model.predict_on_batch(x_batch)
        for pred in predictions:
            confidence = low_confidence(pred[0])
            low_conf.append(confidence.tolist())
    return low_conf


def predict_new(model, x_size=10000, batch_size=64, test_set=test_set):
    """
    本函数调用model.predict预测，速度it/s
    """

    low_conf = []
    for start in tqdm(range(0, len(test_set), x_size)):
        x_batch = []
        end = min(start + x_size, len(test_set))
        new_batch = test_set[start:end]
        for jet in new_batch:
            if cv2.imread(jet) is None:
                img = np.zeros((299, 299, 3))
            else:
                img = cv2.imread(jet)[:, :, ::-1]
                img = cv2.resize(img, (299, 299))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255.
        
        predictions = model.predict(x_batch, batch_size=batch_size)
        for pred in predictions:
            confidence = low_confidence(pred[0])
            low_conf.append(confidence.tolist())
    
    return low_conf


def current_time():
    current_time = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return current_time


if __name__ == '__main__':
#    del_unopen(test_set, ft=0)

    flag = 1
    if flag:
        from keras.models import load_model
        model = load_model('/home/abc/pzw/files/class_22/train0228/models_and_logs/m5032_dn169_v1_l.h5')
        print 'testing model:', model.predict(np.zeros((1, 224, 224, 3)))

        print 'current time is', current_time()
        #low_conf = predict_new(x_size=10000, batch_size=64)
        low_conf = predict_on_batch(model, batch_size=64)
        print 'current time is', current_time()

        test_set_prefix = [test_set[i].split('/')[-1] for i in range(len(test_set))]
        if not os.path.exists('results/json'):
            os.makedirs('results/json')

        # 图片名和对应的类别存入到字典里，保存json
        with open('results/json/' + test_dir.split('/')[-1] + '_lf' + '.json', 'w') as fil:
            json.dump(dict(zip(test_set_prefix, low_conf)), fil)

        print 'saved successfully!!!'


