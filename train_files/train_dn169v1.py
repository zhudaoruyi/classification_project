# -*- coding: utf-8 -*-

import os
from os.path import join
from generator import *
from keras.models import Model
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from keras.applications import InceptionV3, Xception, InceptionResNetV2, \
    DenseNet121, DenseNet169, DenseNet201, \
    NASNetLarge, NASNetMobile, \
    ResNet50
from model import resnet
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard


def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 65:
        lr = 1e-4
    elif epoch > 50:
        lr = 0.5e-5
    elif epoch > 45:
        lr = 1e-4
    elif epoch > 30:
        lr = 1e-3
    print 'Learning rate:', lr
    return lr


def get_model(MODEL, width, height):
    w = 1
    if w:
        base_model = MODEL(weights='imagenet', include_top=False, input_shape=(width, height, 3))
    else:
        base_model = MODEL(weights=None, include_top=False, input_shape=(width, height, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(18, activation='sigmoid')(x)

    for layer in base_model.layers:
        layer.trainable = True

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def train(epochs, batch_size, width, height, prefix='_01', save_dir='models_and_logs/'):
    # model = get_model(InceptionV3, width, height)
#    model = get_model(ResNet50, width, height)
    model = get_model(DenseNet169, width, height)
    # model = get_model(Xception, width, height)
#    model = get_model(InceptionResNetV2, width, height)
    # model = get_model(NASNetMobile, width, height)
#    model = resnet.ResnetBuilder.build_resnet_50((3, width, height), 16)

    model.compile(optimizer=SGD(lr=lr_schedule(0), momentum=0.9, nesterov=True), loss=['binary_crossentropy'], metrics=['accuracy'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    csvlogger = CSVLogger(save_dir + 'log' + str(epochs) + str(batch_size) + prefix + '.log', append=True)
    model_check = ModelCheckpoint(save_dir + 'm' + str(epochs) + str(batch_size) + prefix + '_p.h5', monitor='loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    tblogger = TensorBoard(log_dir=save_dir + 'tblogger' + prefix, histogram_freq=0, write_graph=True, write_images=True)

    model.fit_generator(data_generator(width, height, batch_size, train=True),
                        steps_per_epoch=np.ceil(len(train_dirs)/batch_size),
                        epochs=epochs,
                        validation_data=data_generator(width, height, batch_size, train=False),
                        validation_steps=np.ceil(len(valid_dirs)/batch_size),
                        verbose=1,
                        workers=8,
                        max_q_size=48,
                        callbacks=[csvlogger, model_check, lr_scheduler, tblogger])

#    model.save_weights(save_dir + 'weight' + str(epochs) + str(batch_size) + prefix + '.h5')
    model.save(save_dir + 'm' + str(epochs) + str(batch_size) + prefix + '_l.h5')


if __name__ == '__main__':
    base_dir = '/home/abc/pzw/data/class_2201/'
    labels_dir = 'labels/'
    images_dir = 'images/'
    
    img_dirs = sorted(glob(join(base_dir, images_dir, '*')))
    print '训练图片总数是：%d' % len(img_dirs)

    train_dirs, valid_dirs = train_test_split(img_dirs, test_size=0.1, random_state=42)
    print '训练集图片数量为%d' % len(train_dirs), '验证集图片数量为%d' % len(valid_dirs)

    train(50, 32, 224, 224, prefix='_dn169_v1')

