#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import re
import os
import cv2
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from keras.models import load_model


def set_gpu(gpu_memory_frac=0.8):
    import tensorflow as tf
    import keras.backend as K

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac    # 不全部占满显存, 按给定值分配
    # config.gpu_options.allow_growth=True   # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    K.set_session(sess)


# 单类 18
classes = ['LINE_CHART', 'AREA_CHART', 'BAR_CHART',
           'COLUMN_CHART', 'PIE_CHART', 'UNKNOWN',
           'GRID_TABLE', 'LINE_TABLE', 'QR_CODE',
           'INFO_GRAPH', 'TEXT', 'CANDLESTICK_CHART',
           'PHOTOS', 'SCATTER_CHART', 'RADAR_CHART',
           'DONUT_CHART','LINE_POINT_CHART','DISCRETE_PLOT']

# 设置GPU使用率
#set_gpu(1)

# 导入模型
#model = load_model('/home/abc/pzw/files/class_22/train0228/models_and_logs/m5032_dn169_v1_l.h5')
model = load_model('/home/abc/pzw/files/class_22/train0306/models_and_logs/m5032_dn169_v1_l.h5')


# 被测文件夹
test_dir = 'data/class_2201/'
#test_dir = 'data/data0206/'

# 被测文件夹下的文件夹名
obj_dirs = ['AREA_CHART', 'BAR_CHART', 'COLUMN_CHART',
            'GRID_TABLE', 'LINE_CHART', 'LINE_CHART_and_AREA_CHART',
            'LINE_CHART_and_COLUMN_CHART', 'LINE_TABLE','PIE_CHART',
            'QR_CODE', 'UNKNOWN', 'CANDLESTICK_CHART',
            'TEXT', 'INFO_GRAPH', 'PHOTOS',
            'SCATTER_CHART', 'RADAR_CHART', 'DONUT_CHART',
            'SCATTER_CHART_and_LINE_CHART', 'LINE_POINT_CHART', 'DISCRETE_PLOT',
            'LINE_POINT_CHART_and_COLUMN_CHART']


# 与glob类似功能的函数，只是对图片格式有效
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


def get_label_from_pred(pred, classes=classes):
    """
    :param pred:should be 1D array or list,like [0,0,0,1,0,0,1,0]
    :param classes: default classes=['UNKNOWN', 'LINE_CHART', 'AREA_CHART', 'BAR_CHART',
                                    'COLUMN_CHART', 'PIE_CHART', 'GRID_TABLE', 'LINE_TABLE']
    :return:
    """
    pred_label = []
    index = np.nonzero(np.round(pred))[0]
    for i in index:
        pred_label.append(classes[i])
    return pred_label


def label_predict(chart_type, img_dirs):
    wrong_classify_dict = {}
    are_n, bar_n, col_n, gri_n, lin_n, lia_n, lic_n, lit_n, pie_n, qrc_n, unk_n, can_n, txt_n, \
    inf_n, phs_n, sca_n, rad_n, don_n, scl_n, lpchart_n, dp_n, lpcol_n,  = 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i, key in tqdm(enumerate(sorted(img_dirs))):
        img = cv2.imread(key)[:, :, ::-1]
#        img_arr = cv2.resize(img, (299, 299)) / 255.
        img_arr = cv2.resize(img, (224, 224)) / 255.
        y_pred_single = model.predict(np.expand_dims(img_arr, axis=0))
        pred_label = get_label_from_pred(y_pred_single[0])
        if pred_label == ['AREA_CHART']:
            are_n += 1
        if pred_label == ['BAR_CHART']:
            bar_n += 1
        if pred_label == ['COLUMN_CHART']:
            col_n += 1
        if pred_label == ['GRID_TABLE']:
            gri_n += 1
        if pred_label == ['LINE_CHART']:
            lin_n += 1
        if pred_label == ['LINE_CHART', 'AREA_CHART']:
            lia_n += 1
        if pred_label == ['LINE_CHART', 'COLUMN_CHART']:
            lic_n += 1
        if pred_label == ['LINE_TABLE']:
            lit_n += 1
        if pred_label == ['PIE_CHART']:
            pie_n += 1
        if pred_label == ['QR_CODE']:
            qrc_n += 1
        if pred_label == ['UNKNOWN']:
            unk_n += 1
        if pred_label == ['CANDLESTICK_CHART']:
            can_n += 1
        if pred_label == ['TEXT']:
            txt_n += 1
        if pred_label == ['INFO_GRAPH']:
            inf_n += 1
        if pred_label == ['PHOTOS']:
            phs_n += 1
        if pred_label == ['SCATTER_CHART']:
            sca_n += 1
        if pred_label == ['RADAR_CHART']:
            rad_n += 1
        if pred_label == ['DONUT_CHART']:
            don_n += 1
        if pred_label == ['LINE_POINT_CHART']:
            lpchart_n += 1
        if pred_label == ['DISCRETE_PLOT']:
            dp_n += 1
        if pred_label == ['LINE_CHART', 'SCATTER_CHART']:
            scl_n += 1
        if pred_label == ['COLUMN_CHART', 'LINE_POINT_CHART']:
            lpcol_n += 1

        if pred_label != chart_type:
            wrong_classify_dict[key.split('/')[-1]] = pred_label
    return are_n, bar_n, col_n, gri_n, lin_n, lia_n, lic_n, lit_n, pie_n, qrc_n, unk_n, can_n, txt_n, inf_n, phs_n, sca_n, rad_n, don_n, lpchart_n, dp_n, scl_n, lpcol_n, wrong_classify_dict


start = time.clock()

for ids, direcs in enumerate(obj_dirs):
    test_set = list_images(test_dir + direcs)
    print 'sum of images in %s directory to be predict: ' % direcs, len(test_set)
    if len(test_set):
        chart_type = [direcs]
        if direcs == 'LINE_CHART_and_AREA_CHART':
            chart_type = ['LINE_CHART', 'AREA_CHART']
        elif direcs == 'LINE_CHART_and_COLUMN_CHART':
            chart_type = ['LINE_CHART', 'COLUMN_CHART']
        elif direcs == 'SCATTER_CHART_and_LINE_CHART':
            chart_type = ['LINE_CHART', 'SCATTER_CHART']
        elif direcs == 'LINE_POINT_CHART_and_COLUMN_CHART':
            chart_type = ['COLUMN_CHART', 'LINE_POINT_CHART']

        are_n, bar_n, col_n, gri_n, lin_n, lia_n, lic_n, lit_n, pie_n, qrc_n, unk_n, can_n, txt_n, inf_n, \
        phs_n, sca_n, rad_n, don_n, lpchart_n, dp_n, scl_n, lpcol_n, wr = label_predict(chart_type, test_set)

        print str(direcs) + ' classification results: ', '\n'
        print 'are %d,'%are_n, 'bar %d,'%bar_n, 'col %d,'%col_n, 'grit %d,'%gri_n, 'lich %d,'%lin_n, 'li_ar %d,'%lia_n,\
              'li_col %d,'%lic_n, 'lita %d,'%lit_n, 'pie %d,'%pie_n, 'QRcode %d,'%qrc_n, 'unk %d,'%unk_n, 'Cansti %d,'%can_n, \
              'Text %d,'%txt_n, 'Infgra %d,'%inf_n, 'photos %d,'%phs_n, 'sca %d,'%sca_n, 'radar %d,'%rad_n, 'donut %d'%don_n, \
              'scali %d,'%scl_n, 'lipo %d,'%lpchart_n, 'discrep %d,'%dp_n, 'lipo column %d,'%lpcol_n
#        print 'wrong classified:%s'%wr
        save_dir = 'evaluate_results/' + test_dir.split('/')[-2] + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(save_dir + direcs + '.json', 'w') as f:
            json.dump(wr, f)

        print '-'*50

