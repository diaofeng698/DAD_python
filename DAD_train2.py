import cv2
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import wandb
from wandb.keras import WandbCallback

'''
**************************************************
Operating Instructions:
1 Configure wandb project name
2 Configure batch_size, training epochs, stopping_patience, classes in wandb
3 Configure weights_folder for saving model weights
4 Configure driver_file, dataset_folder for reading datasets and labels
5 Configure map for classification and label mapping
6 Configure driver_valid_list  driver_test_list for dataset split
***************************************************
'''

if __name__ == '__main__':
    root = os.getcwd()
# __________________________Wandb Online Version Train Visualization_________________________

# 1 Configure wandb project name
    wandb.init(project="MobileNet_DAD_1")

# 2 Configure learning rate, batch_size, training epochs, stopping_patience, classes
    config = wandb.config
    config.learning_rate = 0.0015
    config.batch_size = 32
    config.epochs = 100
    config.stopping_patience = 10

# 3 Configure weights_folder for saving model weights
    weights_folder = 'DAD_weights_2'

    save_weights_path = os.path.join(root, weights_folder)
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    print('save model path:', save_weights_path)

# 4 Configure driver_file, dataset_folder for reading datasets and labels
    dataset_folder = 'train_dataset_1'
    dataset_route = os.path.join(root, dataset_folder)

# 5 Configure map_list for classification and label mapping
    index_to_class = {0: 'safe_driving', 1: 'eating', 2: 'drinking', 3: 'smoking',
                      4: 'phone_interaction', 5: 'other_activity'}
    class_to_index = {'safe':0, 'eat':1, 'drink':2, 'smoke':3, 'phone':4, 'other':5}

    train_image = []
    classes_list = os.listdir(dataset_route)

    for class_name in classes_list:
        print(f'now we are in the folder {class_name}')
        imgs_folder_path = os.path.join(root, dataset_folder, class_name)
        imgs_list = os.listdir(imgs_folder_path)

        for img_name in tqdm(imgs_list):
            img_path = os.path.join(imgs_folder_path, img_name)
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (224, 224))
            img = np.repeat(img[..., np.newaxis], 3, -1)
            label = class_to_index[class_name]
            driver = img_name.split('_')[0]
            train_image.append([img, label, driver])

    print('total images:', len(train_image))
    save_img_name = index_to_class[train_image[-1][1]] + '_driver_' + train_image[-1][-1] + '.jpg'
    cv2.imwrite(save_img_name, train_image[-1][0])


# ______________________Splitting the train, valid and test dataset_________________________

# 6 Configure driver_valid_list  driver_test_list for dataset split
    random.shuffle(train_image)
    driver_valid_list = {'tinghao'}
    driver_test_list = {}

    X_train, y_train = [], []
    X_valid, y_valid = [], []
    X_test, y_test = [], []

    for image, label, driver in train_image:
        if driver in driver_test_list:
            X_test.append(image)
            y_test.append(label)
        elif driver in driver_valid_list:
            X_valid.append(image)
            y_valid.append(label)
        else:
            X_train.append(image)
            y_train.append(label)

    X_train = np.array(X_train).reshape(-1, 224, 224, 3)
    X_valid = np.array(X_valid).reshape(-1, 224, 224, 3)
    X_test_array = np.array(X_test).reshape(-1, 224, 224, 3)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_valid shape: {X_valid.shape}')
    print(f'X_test shape: {X_test_array.shape}')

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test_array = np.array(y_test)
    print(f'y_train shape: {y_train.shape}')
    print(f'y_valid shape: {y_valid.shape}')
    print(f'y_test shape: {y_test_array.shape}')

# ___________________________Build Model_____________________________________

    base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(config.classes, activation='softmax')(x)  # final layer with softmax activation
    model = tf.keras.Model(inputs=base_model.input, outputs=prediction)
    #print(model.summary())

# ___________________________Model Train_______________________________________

    opt = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[acc])
    checkpointer = ModelCheckpoint(filepath=save_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystopper = EarlyStopping(monitor='val_loss', patience=config.stopping_patience, verbose=1, min_delta=0.001, mode='min')

    datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
    )
    train_data_generator = datagen.flow(X_train, y_train, batch_size=config.batch_size)

    # Fits the model on batches with real-time data augmentation:
    mobilenet_history = model.fit(train_data_generator, steps_per_epoch=len(X_train) / config.batch_size, callbacks=[checkpointer, earlystopper,WandbCallback()],
                                  epochs=config.epochs, verbose=1, validation_data=(X_valid, y_valid))

# __________________________Train Visualization_________________________

    # Can be replaced by online version
    
    # plt.plot(mobilenet_history.history['loss'])
    # plt.plot(mobilenet_history.history['val_loss'])
    # plt.title('loss vs epochs')
    # plt.ylabel('loss')
    # plt.xlabel('epochs')
    # plt.legend(['training', 'validation'], loc='upper right')
    # plt.savefig('train_loss.jpg')

    # plt.plot(mobilenet_history.history['sparse_categorical_accuracy'])
    # plt.plot(mobilenet_history.history['val_sparse_categorical_accuracy'])
    # plt.title('accuracy vs epochs')
    # plt.ylabel('accuracy')
    # plt.xlabel('epochs')
    # plt.legend(['training', 'validation'], loc='lower right')
    # plt.savefig('train_accuracy.jpg')


# ____________________________ Evaluate on test dataset_______________________________
    loss, acc = model.evaluate(X_test_array, y_test_array)
    print(f'last epoch loss: {loss}')
    print(f'last epoch test accuracy:{acc:.2%}')

    model_load = tf.keras.models.load_model(save_weights_path)
    loss, acc = model_load.evaluate(X_test_array, y_test_array)
    print(f'best loss: {loss}')
    print(f'best test accuracy:{acc:.2%}')





