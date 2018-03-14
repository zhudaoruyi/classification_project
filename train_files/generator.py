from sklearn.model_selection import train_test_split
from os.path import join
from augment import *
from glob import glob
import numpy as np
import threading
import cv2

base_dir = '/home/abc/pzw/data/class_2201/'
labels_dir = 'labels/'
images_dir = 'images/'

# 18 single classes, 4 combine classes
classes = ['LINE_CHART', 'AREA_CHART', 'BAR_CHART',
           'COLUMN_CHART', 'PIE_CHART', 'UNKNOWN',
           'GRID_TABLE', 'LINE_TABLE', 'QR_CODE', 
           'INFO_GRAPH', 'TEXT', 'CANDLESTICK_CHART',
           'PHOTOS','SCATTER_CHART', 'RADAR_CHART',
           'DONUT_CHART', 'LINE_POINT_CHART', 'DISCRETE_PLOT']


def get_label(label_list, classes=classes):
    """
    label_list:['LINE_CHART', 'COLUMN_CHART']
    return:
        [0,1,0,0,1,0,0,0]
    """
    y = np.zeros(len(classes))
    for m, key in enumerate(label_list):
        if len(key):
            y[classes.index(key)] = 1
    return y


img_dirs = sorted(glob(join(base_dir, images_dir, '*')))
train_dirs, valid_dirs = train_test_split(img_dirs, test_size=0.1, random_state=42)


class ThreadsafeIter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    def g(*args, **kw):
        return ThreadsafeIter(f(*args, **kw))
    return g


@threadsafe_generator
def data_generator(width, height, batch_size, train=True):
    """
    input:
        directories of train or validation,eg:train_dirs,valid_dirs
    output:
        yield X,y
    """
    if train:
        data_dirs = train_dirs
    else:
        data_dirs = valid_dirs

    while True:
        for start in range(0, len(data_dirs), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(data_dirs))
            dirs_batch = data_dirs[start:end]
            # print(dirs_batch)
            for i, key in enumerate(dirs_batch):
                # print(key)
                img = cv2.imread(key)[:, :, ::-1]
                img = cv2.resize(img, (width, height))
                if train:
                    img = randomCrop(img, size=(width, height))
                    img = randomHueSaturationValue(img,
                                                   hue_shift_limit=(-50, 50),
                                                   sat_shift_limit=(-5, 5),
                                                   val_shift_limit=(-15, 15))
                    img = randomShiftScaleRotate(img,
                                                 shift_limit=(-0.0001, 0.0001),
                                                 scale_limit=(-0.1, 0.1),
                                                 rotate_limit=(-10, 10))
                    img = randomHorizontalFlip(img)
                    img = randomVerticallyFlip(img)
                x_batch.append(img)

                with open(join(base_dir, labels_dir, key.split('/')[-1] + '.txt'), 'r') as f:
                    img_lab = f.read()
                y = get_label(img_lab.split('\n'))
                y_batch.append(y)
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


if __name__ == '__main__':
    img_dirs = sorted(glob(join(base_dir, images_dir, '*')))

    train_dirs, valid_dirs = train_test_split(img_dirs, test_size=0.1, random_state=42)
    print(len(train_dirs), len(valid_dirs))
