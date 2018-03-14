#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import cv2
import shutil
import datetime
import numpy as np
from tqdm import tqdm
from keras.models import load_model

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
    image_type = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif']
    paths = []
    for file_and_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_and_dir)):
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in image_type:
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


def classify(predictions):
    types = []
    scores = []
    top_k = predictions[0].argsort()[::-1]

    if np.max(predictions[0]) > 0.5:
        # 如果最高的是LINE_CHART
        if top_k[0] == 0:
            line_s = predictions[0][0] = 1  # 折线图得分置为1
            area_s = predictions[0][1]  # 获取面积图得分
            colu_s = predictions[0][3]  # 获取柱子图得分
            line_point_s = predictions[0][16]  # 获取均匀点折线图得分
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][0], predictions[0][1], predictions[0][3], predictions[0][16] = line_s, area_s, colu_s, line_point_s  # 返回折线、面积、柱子图得分

        # 如果最高的是AREA_CHART
        elif top_k[0] == 1:
            area_s = predictions[0][1] = 1  # 面积图得分置为1
            line_s = predictions[0][0]  # 获取折线图得分
            line_point_s = predictions[0][16]  # 获取均匀点折线图得分
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][1], predictions[0][0], predictions[0][16] = area_s, line_s, line_point_s

        # 如果最高的是BAR_CHART
        elif top_k[0] == 2:
            bar_s = predictions[0][2] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][2] = bar_s

        # 如果最高的是COLUMN_CHART
        elif top_k[0] == 3:
            colu_s = predictions[0][3] = 1  # 柱子图得分置为1
            line_s = predictions[0][0]  # 获取折线图得分
            line_point_s = predictions[0][16]  # 获取均匀点折线图得分
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][3], predictions[0][0], predictions[0][16] = colu_s, line_s, line_point_s

        # 如果最高的是PIE_CHART
        elif top_k[0] == 4:
            pie_s = predictions[0][4] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][4] = pie_s

        # 如果最高的是UNKNOWN
        elif top_k[0] == 5:
            unk_s = predictions[0][5] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][5] = unk_s

        # 如果最高的是GRID_TABLE
        elif top_k[0] == 6:
            grid_s = predictions[0][6] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][6] = grid_s

        # 如果最高的是LINE_TABLE
        elif top_k[0] == 7:
            lita_s = predictions[0][7] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][7] = lita_s

        # 如果最高是QR_CODE
        elif top_k[0] == 8:
            qrco_s = predictions[0][8] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][8] = qrco_s

        # 如果最高是INFO_GRAPH
        elif top_k[0] == 9:
            infg_s = predictions[0][9] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][9] = infg_s

        # 如果最高是TEXT
        elif top_k[0] == 10:
            text_s = predictions[0][10] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][10] = text_s

        # 如果最高是CANDLESTICK_CHART
        elif top_k[0] == 11:
            cand_s = predictions[0][11] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][11] = cand_s

        # 如果最高是PHOTOS
        elif top_k[0] == 12:
            phot_s = predictions[0][12] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][12] = phot_s

        # 如果最高是SCATTER
        elif top_k[0] == 13:
            scat_s = predictions[0][13] = 1
            line_s = predictions[0][0]
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][13], predictions[0][0] = scat_s, line_s

        # 如果最高是RADAR_CHART
        elif top_k[0] == 14:
            rada_s = predictions[0][14] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][14] = rada_s

        # 如果最高是DONUT_CHART
        elif top_k[0] == 15:
            donu_s = predictions[0][15] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][15] = donu_s

        # 如果最高是LINE_POINT_CHART
        elif top_k[0] == 16:
            line_point_s = predictions[0][16] = 1
            line_s = predictions[0][0]
            area_s = predictions[0][1]  # 获取面积图得分
            colu_s = predictions[0][3]  # 获取柱子图得分
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][16], predictions[0][0], predictions[0][1], predictions[0][3] = line_point_s, line_s, area_s, colu_s

        # 如果最高是DISCRETE_PLOT
        elif top_k[0] == 17:
            dipl_s = predictions[0][17] = 1
            predictions[0] = np.zeros((len(predictions[0])))  # 所有类别预测值置为0
            predictions[0][17] = dipl_s

        for node_id in top_k:
            score = predictions[0][node_id]
            if score > 0.8:
#                types.append(Label_Line[node_id])
                types.append(chart_type_dic[node_id])
                scores.append(score)
        if len(types) == 0:
            # 所有类型的概率都很低, 那就选概率最高的
            node_id = top_k[0]
            score = predictions[0][node_id]
            if score > 0.5:
#                types.append(Label_Line[node_id])
                types.append(chart_type_dic[node_id])
            else:
                types.append("UNKNOWN")
            scores.append(score)
    else:
        types.append("OTHER_MEANINGFUL")
        scores.append(1 - np.sum(predictions[0]) / len(predictions[0]))

#    result = [[t, float(s)] for (t, s) in zip(types, scores)]
    result = [t for t in types]
    return result


def del_unopen(test_set):
    print '删除打不开的图片...'
    for i, k in tqdm(enumerate(test_set)):
        if cv2.imread(k) is None:
            del (test_set[i])
            os.remove(k)
    print '现在需要分类的图片总数是：%d' % len(test_set)


def predict(test_set, model, batch_size=64):
    x = []
    for i, k in tqdm(enumerate(test_set)):
        img = cv2.imread(k)[:, :, ::-1]
        img = cv2.resize(img, (299, 299))
        x.append(np.array(img)/255.)
    x = np.array(x)
    print 'shape of x:', x.shape
    print 'starting predicting...'
    print 'start time is :', current_time()
    predictions = model.predict(x, batch_size=batch_size)
    print 'end time is :', current_time()
    return predictions


def predict_on_batch(test_set, model, batch_size=64):
    predictions = np.zeros((len(test_set), len(classes)))
    print 'start time is :', current_time()
    for start in tqdm(range(0, len(test_set), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(test_set))
        new_batch = test_set[start:end]
        for img_dir in new_batch:
            img = cv2.imread(img_dir)[:, :, ::-1]
#            img = cv2.resize(img, (299, 299))
            img = cv2.resize(img, (224, 224))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255.
        batch_pred = model.predict_on_batch(x_batch)
        predictions[start:end] = batch_pred
    print 'end time is :', current_time()
    return np.array(predictions)


def move_image_to_class(test_set, predictions, dst_dirs = 'results/lf_classify_da011001/'):
    # 统计一下数据，低概率的图片，单类的图片，双类的图片，三类的图片
    low_prob_sum, single_sum, double_sum, triple_sum = 0, 0, 0, 0

    print 'start time is :', current_time()
    for ids, pred in enumerate(predictions):
        pred_label = classify([pred])
        pred_label = sorted(pred_label, reverse=True)

        # dst_dirs = 'results/' + 'lf_classify_010802/'
        # dst_dirs = '/home/zhwpeng/data/images0109/' + 'wechat_results/'
        if not os.path.exists(dst_dirs):
            os.mkdir(dst_dirs)

        # 低概率图片
        if pred_label == []:
            if not os.path.exists(dst_dirs + 'low_prob'):
                os.mkdir(dst_dirs + 'low_prob/')
            shutil.copy(test_set[ids], dst_dirs + 'low_prob/')
            low_prob_sum += 1
        # 单类别的图片
        if pred_label != [] and len(pred_label) == 1:
            if not os.path.exists(dst_dirs + pred_label[0]):
                os.mkdir(dst_dirs + pred_label[0])
            shutil.copy(test_set[ids], dst_dirs + pred_label[0])
            single_sum += 1
        # 双类别的图片
        if pred_label != [] and len(pred_label) == 2:
            if not os.path.exists(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1])):
                os.mkdir(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1]))
            shutil.copy(test_set[ids], dst_dirs + str(pred_label[0] + '_and_' + pred_label[1]))
            double_sum += 1
        # 三类别的图片
        if pred_label != [] and len(pred_label) == 3:
            if not os.path.exists(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1] + '_and_' + pred_label[2])):
                os.mkdir(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1] + '_and_' + pred_label[2]))
            shutil.copy(test_set[ids], dst_dirs + str(pred_label[0] + '_and_' + pred_label[1] + '_and_' + pred_label[2]))
            triple_sum += 1
    print '低概率图片数量是：%d' % low_prob_sum, '单类图片的数量是：%d' % single_sum, \
        '双类图片的数量是：%d' % double_sum, '三类别图片的数量是：%d' % triple_sum
    print 'end time is :', current_time()
    print 'classify successfully!'
    print '-'*30, '\n', '-'*30


if __name__ == '__main__':
    # 导入模型
#    model = load_model('/home/abc/pzw/files/class_22/train0228/models_and_logs/m5016_ir_v1_l.h5')
    model = load_model('/home/abc/pzw/files/class_22/train0306/models_and_logs/m5032_dn169_v1_l.h5')
    # 需要分类的数据集
#    test_dir = 'data/data030601'
    test_dir = 'data/0308'
#    test_dir = 'pickouts/results0211_ct'
    test_set = sorted(list_images(test_dir))
    print '需要分类的图片总数是：%d' % len(test_set)
#    del_unopen(test_set)
    # 预测图片集的类别
    # predictions = predict(test_set, model, batch_size=4)
    predictions = predict_on_batch(test_set, model, batch_size=64)

    # 移动图片到相应的类别文件夹
    move_image_to_class(test_set, predictions, dst_dirs='results/classified0308/')

